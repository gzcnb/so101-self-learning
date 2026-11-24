"""Park an empty arm from a logged, bounded recovery pose."""

import argparse
import json

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from hover_probe import CHECKS, LAB, PORT, positions, read
from learn_orange_grasp import AXES, RETURN_TRANSFER
from orange_approach_probe import GRIPPER, hold
from swap_blocks import APERTURE, IDLE_JAW, SwapTransfer


class HighRetreat(SwapTransfer):
    minimum_x = -38


class SagRetreat(HighRetreat):
    minimum_x = -96
    maximum_elbow = 2850
    step_counts = 24


def main(run_id, sagged=False, far_return=False):
    if not run_id.isascii() or not run_id.replace('_', '').isalnum():
        raise ValueError('Invalid run ID')
    from settings import require_hardware_enabled
    require_hardware_enabled()
    load = lambda path: json.loads((CHECKS / path).read_text())
    previous = load(f'{run_id}/report.json')
    reached = {stage['name'] for stage in previous['poses']}
    before_pick = ('clearance' in reached and 'source_hover' not in reached and 'closed' not in previous['vision']
                   and previous.get('grasp_check') is None)
    if (sagged or far_return) and not before_pick:
        raise RuntimeError('The sag recovery applies only before a source approach')
    if far_return and 'edge_source_view' not in reached:
        raise RuntimeError('The far retreat requires the logged edge inspection')
    if ('return_to_source' not in reached and not before_pick) or any(previous['final_torque'].values()):
        raise RuntimeError('A known empty gripper and all six disabled torques are required')
    directory = CHECKS / f'{run_id}_park_open'
    if sagged and directory.exists():
        directory = directory.with_name(f'{run_id}_park_sag')
    if far_return:
        directory = directory.with_name(f'{run_id}_park_far')
    directory.mkdir(mode=0o700, exist_ok=False)
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id='so101-workbench', calibration_dir=LAB / 'calibration'))
    bus = robot.bus
    record = {'poses': [], 'vision': {}, 'planned_aperture': APERTURE}
    bus.connect()
    try:
        start = positions(bus, bus.motors)
        previous_start = previous['released_positions']
        aperture = IDLE_JAW if before_pick else APERTURE
        if not robot.is_calibrated or abs(start[GRIPPER] - aperture) > 9:
            raise RuntimeError('Parking requires a known empty gripper and valid calibration')
        for axis, value in start.items():
            tolerance = (12 if before_pick else 125) if axis == 'shoulder_lift' else 24 if axis == 'elbow_flex' else 8
            if far_return and axis in ('shoulder_lift', 'shoulder_pan'):
                tolerance = 70 if axis == 'shoulder_lift' else 25
            if sagged and axis == 'elbow_flex':
                if not 155 <= value - previous_start[axis] <= 205 or read(bus, 'Torque_Enable', axis):
                    raise RuntimeError('Elbow does not match the checked sag recovery')
                continue
            if abs(value - previous_start[axis]) > tolerance or read(bus, 'Torque_Enable', axis):
                raise RuntimeError('Current joint readings disagree with the recorded rollback')
        models = [load(f'{name}.json')['encoder_zeros'] for name in ('base_alignment_p4_checked', 'base_alignment_checked')]
        models.append({joint: (entry.range_min + entry.range_max) / 2 for joint, entry in robot.calibration.items()})
        controller = SagRetreat if sagged else HighRetreat if before_pick else SwapTransfer
        task = controller(bus, start, load('base_alignment_p4_verified.json')['raw_positions'], models,
                          directory, record, leg=previous['plan'])
        height = float(task.geometry(start)[:, 2].min())
        lower, upper = (-12, 18) if sagged else (35, 90) if before_pick else (-15, 50)
        if far_return:
            joints = tuple(start[axis] for axis in AXES)
            bounds = ((1902, 1938), (1838, 1910), (2624, 2668), (1668, 1710))
            if any(not lo <= value <= hi for value, (lo, hi) in zip(joints, bounds, strict=True)):
                raise RuntimeError('The joints disagree with the checked far inspection return')
            lower, upper = 29, 64
        if not lower <= height <= upper:
            raise RuntimeError('The arm is outside the validated rollback height range')
        task.floor = 30 if far_return else 33 if before_pick and not sagged else min(0, height - 1.5)
        task.validate(start)
        if sagged:
            last_height = height
            target = start | {'elbow_flex': previous_start['elbow_flex'] - 34}
            for elbow in range(start['elbow_flex'], target['elbow_flex'] - 1, -1):
                sample = target | {'elbow_flex': elbow}
                task.validate(sample)
                next_height = float(task.geometry(sample)[:, 2].min())
                if next_height + 0.01 < last_height:
                    raise RuntimeError('The recovery does not continuously raise the tool')
                last_height = next_height
        with hold(bus, start):
            if far_return:
                shoulder = start | {'shoulder_lift': 1827}
                raised = task.go(tuple(shoulder[axis] for axis in AXES), 30, 'raise_shoulder', corrections=False)
                tucked = raised | {'wrist_flex': 1624}
                task.go(tuple(tucked[axis] for axis in AXES), 39, 'clear_wrist', corrections=False)
                task.go((1970, 1855, 2630, 1630), 39, 'far_return_midpoint', corrections=False)
                task.go(RETURN_TRANSFER, 34, 'far_return_transfer', corrections=False)
            if sagged:
                raised = task.go(tuple(target[axis] for axis in AXES), task.floor, 'raise_elbow', corrections=False)
                if task.geometry(raised)[:, 2].min() < 33:
                    raise RuntimeError('The elbow did not achieve the required retreat height')
                clear_wrist = raised | {'wrist_flex': 1680}
                task.go(tuple(clear_wrist[axis] for axis in AXES), 33, 'clear_wrist', corrections=False)
                task.minimum_x = HighRetreat.minimum_x
                task.maximum_elbow = SwapTransfer.maximum_elbow
            if not before_pick:
                task.photo('before')
                task.raise_clear('raise_open')
                task.go(RETURN_TRANSFER, 31, 'clear_of_source')
                task.photo('source_check')
            task.go((2112, 1880, 2673, 1460), 33 if before_pick else 27, 'above_park')
            task.jaw(IDLE_JAW)
            task.go((2112, 2025, 2673, 1460), 0, 'return_home', corrections=False)
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        try:
            record['final_torque'] = {joint: read(bus, 'Torque_Enable', joint) for joint in bus.motors}
            record['released_positions'] = positions(bus, bus.motors)
        finally:
            bus.disconnect(disable_torque=False)
            (directory / 'report.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps({'error': record.get('error'), 'final_torque': record.get('final_torque'),
                              'stages': [pose['name'] for pose in record['poses']]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--sagged', action='store_true')
    parser.add_argument('--far-return', action='store_true')
    args = parser.parse_args()
    main(args.run_id, args.sagged, args.far_return)
