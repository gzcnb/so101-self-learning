"""Learn and replay a small shoulder lift from raw SO-101 feedback."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from hover_probe import CAMERA, CHECKS, LAB, LIFT_JOINT, PORT, WRIST, positions, read, stop, write
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


GOALS_FROM_CONTACT = (29, 41, 53)
DESIRED_LIFT = 20


def observe(bus, joints, start, floor):
    actual = positions(bus, joints)
    for joint, value in actual.items():
        lower = floor if joint == LIFT_JOINT else start[joint] - 6
        upper = start[joint] + (3 if joint == LIFT_JOINT else 6)
        if not lower <= value <= upper:
            raise RuntimeError(f'{joint} went outside the lift window: {value}')
    if read(bus, 'Status', LIFT_JOINT):
        raise RuntimeError('Shoulder servo reported a hardware fault')
    return actual


def stable_position(observe_now, trace):
    deadline = time.monotonic() + 3.6
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        actual = observe_now()
        value = actual[LIFT_JOINT]
        if not trace or value != trace[-1]:
            stable_since = time.monotonic()
            trace.append(value)
        if time.monotonic() - stable_since >= 0.6:
            return actual
        time.sleep(0.045)
    raise RuntimeError(f'Shoulder did not settle. Encoder history {trace}')


def capture(directory, stage, observe_now):
    shared = os.environ.get('SO101_FRAME_DIR')
    if shared:
        fresh_after = time.time() + 0.25
        for name in ('fixed', 'wrist'):
            source = Path(shared) / f'{name}.jpg'
            deadline = time.monotonic() + 3.5
            while time.monotonic() < deadline:
                observe_now()
                if source.exists() and source.stat().st_mtime >= fresh_after:
                    frame = source.read_bytes()
                    if frame.startswith(b'\xff\xd8') and frame.endswith(b'\xff\xd9'):
                        (directory / f'{stage}_{name}.jpg').write_bytes(frame)
                        break
                time.sleep(0.05)
            else:
                raise RuntimeError(f'No fresh shared {name} frame')
        return
    for name, device in (('fixed', CAMERA), ('wrist', WRIST)):
        path = directory / f'{stage}_{name}.jpg'
        command = ['v4l2-ctl', '--device', device, '--set-fmt-video=width=640,height=480,pixelformat=MJPG',
                   '--stream-mmap=4', '--stream-skip=12', '--stream-count=1', f'--stream-to={path}']
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
            try:
                deadline = time.monotonic() + 2.5
                while process.poll() is None and time.monotonic() < deadline:
                    observe_now()
                    time.sleep(0.045)
                if process.poll() != 0 or not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f'Could not capture {path.name}')
            finally:
                if process.poll() is None:
                    process.kill()


def trial(bus, observe_now, start, goal, index, directory, report):
    before = observe_now()
    row = {'goal': goal, 'before': before, 'lift_trace': [], 'return_trace': []}
    report['trials'].append(row)
    write(bus, 'Goal_Position', LIFT_JOINT, goal)
    try:
        if read(bus, 'Goal_Position', LIFT_JOINT) != goal:
            raise RuntimeError('Cannot verify shoulder goal')
        row['reached'] = stable_position(observe_now, row['lift_trace'])
        capture(directory, f'trial_{index}', observe_now)
        row['at_photos_end'] = observe_now()
        row['lift_counts'] = before[LIFT_JOINT] - row['at_photos_end'][LIFT_JOINT]
        row['reward'] = -abs(DESIRED_LIFT - row['lift_counts'])
    finally:
        observe_now()
        write(bus, 'Goal_Position', LIFT_JOINT, start[LIFT_JOINT])
        row['returned'] = stable_position(observe_now, row['return_trace'])
        if abs(row['returned'][LIFT_JOINT] - start[LIFT_JOINT]) > 8:
            raise RuntimeError('Unable to reset to the local starting window')


def learn(bus, joints, start, goals, directory, report):
    floor = min(goals) - 4

    def observe_now():
        return observe(bus, joints, start, floor)

    settings = {(joint, reg): read(bus, reg, joint) for joint in joints for reg in ('Goal_Velocity', 'Acceleration')}
    cleanup_errors = []
    try:
        for joint in joints:
            write(bus, 'Goal_Velocity', joint, 25)
            write(bus, 'Acceleration', joint, 5)
            write(bus, 'Goal_Position', joint, start[joint])
            if read(bus, 'Goal_Position', joint) != start[joint]:
                raise RuntimeError(f'Initial goal could not be verified for {joint}')
        if any(abs(value - start[joint]) > 2 for joint, value in positions(bus, joints).items()):
            raise RuntimeError('Arm moved before torque enable')
        for joint in joints:
            write(bus, 'Torque_Enable', joint, 1)
        if any(read(bus, 'Torque_Enable', joint) != 1 for joint in joints):
            raise RuntimeError('Failed to torque-enable all joints')
        report['baseline_positions'] = stable_position(observe_now, [])
        capture(directory, 'baseline', observe_now)
        for index, goal in enumerate(goals, 1):
            trial(bus, observe_now, start, goal, index, directory, report)
            row = report['trials'][-1]
            if row['lift_counts'] < 0:
                raise RuntimeError('The shoulder moved toward the table during a lift')
        best = max(report['trials'], key=lambda row: (row['reward'], row['goal']))
        report['selected_goal'] = best['goal']
        trial(bus, observe_now, start, best['goal'], len(goals) + 1, directory, report)
        replay = report['trials'][-1]
        report['encoder_reward_repeated'] = replay['reward'] >= -5 and abs(replay['lift_counts'] - best['lift_counts']) <= 4
    finally:
        for joint in joints:
            try:
                write(bus, 'Torque_Enable', joint, 0)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        for (joint, reg), value in settings.items():
            try:
                write(bus, reg, joint, value)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            raise RuntimeError(f'Cleanup errors: {cleanup_errors}')
