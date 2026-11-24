"""Board-side bounded inspection from a high pose and a known parking point."""

import argparse
import json
import signal

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from hover_probe import CHECKS, LAB, PORT, positions, read, stop
from learn_lift import capture
from learn_orange_grasp import AXES, RETURN_TRANSFER
from orange_approach_probe import hold
from swap_blocks import APERTURE, IDLE_JAW, SwapTransfer


PARK = (2112, 2025, 2673, 1460)
# A folded arm above the left edge leaves the paper visible in the fixed camera.
VIEWS = ((2130, 1808, 2558, 1783), RETURN_TRANSFER)


class Inspection(SwapTransfer):
    minimum_x = -43


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--recover-open', action='store_true')
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    from settings import require_hardware_enabled
    require_hardware_enabled()
    load = lambda path: json.loads((CHECKS / path).read_text())
    directory = CHECKS / args.run_id
    directory.mkdir(mode=0o700, exist_ok=False)
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id='so101-workbench', calibration_dir=LAB / 'calibration'))
    bus = robot.bus
    report = {'poses': [], 'vision': {}, 'planned_aperture': 2460}
    bus.connect()
    try:
        start = positions(bus, bus.motors)
        report['initial'] = start
        jaw_range = (APERTURE - 8, APERTURE + 8) if args.recover_open else (2070, 2145)
        if not robot.is_calibrated or not jaw_range[0] <= start['gripper'] <= jaw_range[1]:
            raise RuntimeError('Inspection needs the expected empty jaw and matching calibration')
        if any(read(bus, 'Torque_Enable', joint) or read(bus, 'Operating_Mode', joint) for joint in start):
            raise RuntimeError('Inspection needs torque off and position mode')
        models = [load('base_alignment_p4_checked.json')['encoder_zeros'], load('base_alignment_checked.json')['encoder_zeros']]
        models.append({joint: (entry.range_min + entry.range_max) / 2 for joint, entry in robot.calibration.items()})
        task = Inspection(bus, start, load('base_alignment_p4_verified.json')['raw_positions'], models,
                          directory, report, leg={'color': 'orange'})
        raw = tuple(start[axis] for axis in AXES)
        start_height = float(task.geometry(start)[:, 2].min())
        if start_height < -9 or not 2035 <= raw[0] <= 2140:
            raise RuntimeError('The arm is outside the checked inspection starting region')
        if args.recover_open:
            bounds = ((2063, 2087), (2174, 2206), (2442, 2470), (1654, 1682))
            if any(not lo <= value <= hi for value, (lo, hi) in zip(raw, bounds, strict=True)):
                raise RuntimeError('Open-gripper recovery requires the checked reboot pose')
        departure = raw[:1] + ((2000 if args.recover_open else 1808),) + raw[2:]
        above_park = PARK[:1] + (1880,) + PARK[2:]
        for waypoint in (departure, *VIEWS, above_park, PARK):
            task.validate(start | dict(zip(AXES, waypoint)))
        rose = False
        with hold(bus, start):
            try:
                task.go(departure, min(0, start_height - 1), 'rise', corrections=False)
                if task.geometry(task.observe())[:, 2].min() < 32:
                    raise RuntimeError('Camera inspection did not establish clearance')
                rose = True
                capture(directory, 'scene_2_a', task.observe)
                capture(directory, 'scene_2_b', task.observe)
                for i, target in enumerate(VIEWS):
                    task.go(target, 30, f'view_{i}', corrections=False)
                    capture(directory, f'scene_{i}_a', task.observe)
                    capture(directory, f'scene_{i}_b', task.observe)
            finally:
                if rose:
                    task.go(above_park, 27, 'above_park', corrections=False)
                    task.jaw(IDLE_JAW)
                    task.go(PARK, 0, 'park', corrections=False)
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        try:
            report['released_positions'] = positions(bus, bus.motors)
            report['final_torque'] = {joint: read(bus, 'Torque_Enable', joint) for joint in bus.motors}
        finally:
            bus.disconnect(disable_torque=False)
            (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
