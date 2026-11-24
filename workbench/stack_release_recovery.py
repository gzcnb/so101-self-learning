"""Set down the verified orange cube in place after a guarded stack approach."""

import json
import sys

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from hover_probe import CHECKS, LAB, PORT, positions, read
from learn_orange_grasp import AXES
from orange_approach_probe import GRIPPER, hold
from swap_blocks import APERTURE, IDLE_JAW, SwapTransfer


class StackRecovery(SwapTransfer):
    minimum_x = -76
    step_counts = 24


def main(run_id):
    if not run_id.isascii() or not run_id.replace('_', '').isalnum():
        raise ValueError('Invalid run ID')
    from settings import require_hardware_enabled
    require_hardware_enabled()
    load = lambda path: json.loads((CHECKS / path).read_text())
    previous = load(f'{run_id}/report.json')
    if (not previous['plan'].get('stack') or previous['grasp_check']['reward'] != 1
            or previous.get('released_at_destination') or any(previous['final_torque'].values())):
        raise RuntimeError('Recovery requires a confirmed held orange block and six disabled torques')
    directory = CHECKS / f'{run_id}_set_down'
    directory.mkdir(mode=0o700, exist_ok=False)
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id='so101-workbench', calibration_dir=LAB / 'calibration'))
    bus = robot.bus
    record = {'poses': [], 'vision': {}, 'planned_aperture': APERTURE}
    bus.connect()
    try:
        start = positions(bus, bus.motors)
        limits = ((2208, 2222), (1946, 1970), (2676, 2700), (1780, 1800), (1041, 1051), (2210, 2257))
        joints = (*AXES, 'wrist_roll', GRIPPER)
        if (not robot.is_calibrated or any(read(bus, 'Torque_Enable', joint) or
                not lo <= start[joint] <= hi for joint, (lo, hi) in zip(joints, limits, strict=True))):
            raise RuntimeError('The current pose is outside of this checked recovery')
        models = [load(f'{name}.json')['encoder_zeros'] for name in ('base_alignment_p4_checked', 'base_alignment_checked')]
        models.append({joint: (entry.range_min + entry.range_max) / 2 for joint, entry in robot.calibration.items()})
        task = StackRecovery(bus, start, load('base_alignment_p4_verified.json')['raw_positions'], models,
                             directory, record, leg=previous['plan'])
        height = float(task.geometry(start)[:, 2].min())
        if not -10 <= height <= 8:
            raise RuntimeError('In-place release requires the measured tabletop height')
        task.floor = min(0, height - 0.8)
        task.validate(start)
        with hold(bus, start):
            held = task.photo('held', (232, 337))
            wrist = held['wrist']
            if not wrist or not 275 <= wrist['center_px'][0] <= 367 or held['fixed'] is None:
                raise RuntimeError('The two views do not confirm the expected orange block')
            task.jaw(APERTURE)
            task.photo('released', (232, 344))
            elbow = start | {'elbow_flex': 2535}
            raised = task.go(tuple(elbow[j] for j in AXES), task.floor, 'raise_elbow', corrections=False)
            wrist_goal = raised | {'wrist_flex': 1680}
            raised = task.go(tuple(wrist_goal[j] for j in AXES), 15, 'clear_wrist', corrections=False)
            lift = raised | {'shoulder_lift': 1870}
            task.go(tuple(lift[j] for j in AXES), 33, 'raise_shoulder', corrections=False)
            task.minimum_x = SwapTransfer.minimum_x
            task.go((2112, 1880, 2673, 1460), 34, 'above_park')
            task.jaw(IDLE_JAW)
            task.go((2112, 2025, 2673, 1460), 0, 'return_home', corrections=False)
            task.photo('reset', (232, 350))
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        try:
            record['released_positions'] = positions(bus, bus.motors)
            record['final_torque'] = {j: read(bus, 'Torque_Enable', j) for j in bus.motors}
        finally:
            bus.disconnect(disable_torque=False)
            (directory / 'report.json').write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps({'error': record.get('error'), 'final_torque': record.get('final_torque'),
                              'stages': [pose['name'] for pose in record['poses']]}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
