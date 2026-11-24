"""Collect clearance and gripper geometry for the first orange-block grasp."""

import argparse
from contextlib import contextmanager
import json
import signal
import time

from hover_probe import CHECKS, LAB, LIFT_JOINT, PORT, positions, read, stop, write
from learn_lift import capture, stable_position
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


APPROACH_GOALS = (2028, 1972, 1920)
PILOT_OPEN = 2590
GRIPPER = 'gripper'


@contextmanager
def hold(bus, start):
    from settings import require_hardware_enabled
    require_hardware_enabled()
    registers = ('Goal_Velocity', 'Acceleration')
    settings = {(joint, reg): read(bus, reg, joint) for joint in start for reg in registers}
    errors = []
    try:
        for joint, value in start.items():
            write(bus, 'Goal_Velocity', joint, 80 if joint == GRIPPER else 45)
            write(bus, 'Acceleration', joint, 7)
            write(bus, 'Goal_Position', joint, value)
            if read(bus, 'Goal_Position', joint) != value:
                raise RuntimeError(f'Initial goal mismatch for {joint}')
        if any(abs(value - start[joint]) > 2 for joint, value in positions(bus, start).items()):
            raise RuntimeError('Arm moved before torque enable')
        for joint in start:
            write(bus, 'Torque_Enable', joint, 1)
        if any(read(bus, 'Torque_Enable', joint) != 1 for joint in start):
            raise RuntimeError('Torque could not be verified on each axis')
        yield
    finally:
        for joint in start:
            try:
                write(bus, 'Torque_Enable', joint, 0)
            except Exception as exc:
                errors.append(str(exc))
        for (joint, reg), value in settings.items():
            try:
                write(bus, reg, joint, value)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError(f'Cleanup failed: {errors}')


def probe(bus, start, directory, report):
    expect_jaw_open = False

    def observe():
        actual = positions(bus, start)
        for joint, value in actual.items():
            low, high = start[joint] - 9, start[joint] + 9
            if joint == LIFT_JOINT:
                low, high = min(APPROACH_GOALS) - 4, start[joint] + 3
            if joint == GRIPPER and expect_jaw_open:
                high = PILOT_OPEN + 8
            if not low <= value <= high:
                raise RuntimeError(f'{joint} outside its approach window: {value}')
        if any(read(bus, 'Status', joint) for joint in (LIFT_JOINT, GRIPPER)):
            raise RuntimeError('A servo reported a fault during approach')
        return actual

    def goal(joint, value):
        write(bus, 'Goal_Position', joint, value)
        if read(bus, 'Goal_Position', joint) != value:
            raise RuntimeError(f'Goal mismatch for {joint}')

    def record(stage):
        sample = report['stages'][stage] = {'shoulder_trace': []}
        sample['positions'] = stable_position(observe, sample['shoulder_trace'])
        capture(directory, stage, observe)
        sample['after_photos'] = observe()
        return sample['after_photos']

    def settle_gripper(target):
        deadline = time.monotonic() + 9
        while time.monotonic() < deadline:
            if abs(observe()[GRIPPER] - target) <= 8:
                return
            time.sleep(0.06)
        raise RuntimeError(f'Gripper failed to reach unloaded target {target}')

    with hold(bus, start):
        previous = record('start')[LIFT_JOINT]
        try:
            for index, requested in enumerate(APPROACH_GOALS, 1):
                goal(LIFT_JOINT, requested)
                observed = record(f'lift_{index}')
                if not requested - 4 <= observed[LIFT_JOINT] <= min(requested + 32, previous - 23):
                    raise RuntimeError('Shoulder did not enter the intended lift interval')
                previous = observed[LIFT_JOINT]
            if previous > 1948:
                raise RuntimeError('Shoulder has not established the minimum opening clearance')
            expect_jaw_open = True
            goal(GRIPPER, PILOT_OPEN)
            settle_gripper(PILOT_OPEN)
            opened = record('open')
            if abs(opened[GRIPPER] - PILOT_OPEN) > 20:
                raise RuntimeError('Could not verify the trial aperture')
        finally:
            observe()
            goal(GRIPPER, start[GRIPPER])
            settle_gripper(start[GRIPPER])
            closed = record('close_clear')
            if not start[GRIPPER] - 9 <= closed[GRIPPER] <= start[GRIPPER] + 9:
                raise RuntimeError('Gripper did not return to its starting aperture')
            expect_jaw_open = False
            for requested in (*reversed(APPROACH_GOALS[:-1]), start[LIFT_JOINT]):
                if requested <= observe()[LIFT_JOINT]:
                    continue
                goal(LIFT_JOINT, requested)
                restored = stable_position(observe, [])
                if abs(restored[LIFT_JOINT] - requested) > 22:
                    raise RuntimeError('Shoulder did not return through the lift path')
            record('returned')
