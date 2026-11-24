"""Measure one bounded lift from the photographed fourth tabletop contact."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


LAB = Path(__file__).resolve().parent
CHECKS = LAB / 'checks'
from settings import SERIAL_PORT as PORT, FIXED_CAMERA as CAMERA, WRIST_CAMERA as WRIST
LIFT_JOINT = 'shoulder_lift'
LIFT_COUNTS = 29
MIN_MEASURED_LIFT = 6


def stop(_signum, _frame):
    raise InterruptedError('Hover probe interrupted')


def read(bus, register, joint):
    return int(bus.read(register, joint, normalize=False, num_retry=1))


def write(bus, register, joint, value):
    bus.write(register, joint, value, normalize=False, num_retry=1)


def positions(bus, joints):
    return {joint: read(bus, 'Present_Position', joint) for joint in joints}


def check_travel(bus, joints, start):
    actual = positions(bus, joints)
    for joint in joints:
        lower = start[joint] - (LIFT_COUNTS + 5 if joint == LIFT_JOINT else 5)
        if not lower <= actual[joint] <= start[joint] + 5:
            raise RuntimeError(f'{joint} exceeded bounded test travel: {actual[joint]}')
    return actual


def settle(bus, joints, start, target, stage, report):
    deadline = time.monotonic() + 3.5
    consecutive = 0
    shoulder_trace = []
    report.setdefault('shoulder_trace', {})[stage] = shoulder_trace
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        actual = check_travel(bus, joints, start)
        shoulder = actual[LIFT_JOINT]
        if not shoulder_trace or shoulder != shoulder_trace[-1]:
            shoulder_trace.append(shoulder)
            stable_since = time.monotonic()
        consecutive = consecutive + 1 if abs(shoulder - target) <= 7 else 0
        if consecutive == 3:
            return actual
        if stage == 'raised' and start[LIFT_JOINT] - shoulder >= MIN_MEASURED_LIFT and time.monotonic() - stable_since >= 0.5:
            return actual
        time.sleep(0.035)
    raise RuntimeError(f'Failed to settle near shoulder position {target}: {shoulder_trace}')


def photograph(bus, joints, start, stage):
    for name, camera in (('fixed', CAMERA), ('wrist', WRIST)):
        path = CHECKS / f'hover_p4_{stage}_{name}.jpg'
        command = ['v4l2-ctl', '--device', camera, '--set-fmt-video=width=640,height=480,pixelformat=MJPG',
                   '--stream-mmap=4', '--stream-skip=4', '--stream-count=1', f'--stream-to={path}']
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
            try:
                deadline = time.monotonic() + 2.5
                while process.poll() is None and time.monotonic() < deadline:
                    check_travel(bus, joints, start)
                    time.sleep(0.035)
                if process.poll() != 0 or not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f'Could not capture {stage} {name} photo')
            finally:
                if process.poll() is None:
                    process.kill()


def exercise(bus, joints, start, target, report):
    original_settings = {
        (joint, register): read(bus, register, joint)
        for joint in joints for register in ('Goal_Velocity', 'Acceleration')
    }
    cleanup_errors = []
    try:
        for joint in joints:
            write(bus, 'Goal_Velocity', joint, 25)
            write(bus, 'Acceleration', joint, 5)
            write(bus, 'Goal_Position', joint, start[joint])
            if read(bus, 'Goal_Position', joint) != start[joint]:
                raise RuntimeError(f'Unable to verify the initial goal for {joint}')
        if any(abs(value - start[joint]) > 2 for joint, value in positions(bus, joints).items()):
            raise RuntimeError('Arm moved before torque could be enabled')
        for joint in joints:
            write(bus, 'Torque_Enable', joint, 1)
        if any(read(bus, 'Torque_Enable', joint) != 1 for joint in joints):
            raise RuntimeError('Could not torque-enable all six joints')
        settle(bus, joints, start, start[LIFT_JOINT], 'hold', report)
        photograph(bus, joints, start, 'hold')
        write(bus, 'Goal_Position', LIFT_JOINT, target)
        try:
            if read(bus, 'Goal_Position', LIFT_JOINT) != target:
                raise RuntimeError('Could not verify the lift goal sent to the shoulder')
            report['raised'] = settle(bus, joints, start, target, 'raised', report)
            photograph(bus, joints, start, 'raised')
            report['raised_after_photos'] = check_travel(bus, joints, start)
        finally:
            check_travel(bus, joints, start)
            write(bus, 'Goal_Position', LIFT_JOINT, start[LIFT_JOINT])
            report['returned'] = settle(bus, joints, start, start[LIFT_JOINT], 'returned', report)
    finally:
        for joint in joints:
            try:
                write(bus, 'Torque_Enable', joint, 0)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        for (joint, register), value in original_settings.items():
            try:
                write(bus, register, joint, value)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if cleanup_errors:
            raise RuntimeError(f'Cleanup errors: {cleanup_errors}')
