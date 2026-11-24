"""Run one checked leg of the orange/dark-gray tabletop swap."""

import argparse
import json
import signal
import time

import cv2
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from hover_probe import CHECKS, LAB, PORT, positions, read, stop, write
from learn_lift import capture
from learn_orange_grasp import AXES, RETURN_TRANSFER, GraspEpisode
from orange_approach_probe import GRIPPER, hold
from supervise_snapshot import find_orange_target
from swap_vision import find_dark_parts, match_support_edge


TORQUE_LIMIT = 160
APERTURE = 2460
IDLE_JAW = 2085
CORRECTION_VIEW = (2125, 1833, 2589, 1781)
EDGE_APPROACH = (2160, 1860, 2615, 1680)
EDGE_SOURCE_VIEW = (2125, 1831, 2583, 1781)
SWAP_APPROVAL = {'action': 'swap', 'object_a': 'orange', 'object_b': 'dark_gray_with_ring'}
STACK_APPROVAL = SWAP_APPROVAL | {'action': 'stack'}


def nearby(candidates, target, radius):
    eligible = [(float(np.linalg.norm(np.subtract(item['center_px'], target))), item) for item in candidates]
    eligible = sorted(((dist, item) for dist, item in eligible if dist <= radius), key=lambda value: value[0])
    return eligible[0][1] if eligible else None


class SwapTransfer(GraspEpisode):
    step_counts = 47
    minimum_x = -30
    maximum_elbow = 2800

    def __init__(self, *args, leg):
        super().__init__(*args)
        self.leg = leg
        self.color = leg['color']

    def validate(self, sample):
        limits = ((1890, 2230), (1800, 2500), (1900, self.maximum_elbow), (1370, 1915))
        for name, (lower, upper) in zip(AXES, limits, strict=True):
            if not lower <= sample[name] <= upper:
                raise RuntimeError(f'{name} left the checked swap range: {sample[name]}')
        if abs(sample['wrist_roll'] - self.initial['wrist_roll']) > 12 or not 2060 <= sample[GRIPPER] <= APERTURE + 8:
            raise RuntimeError('Wrist roll or jaw aperture left the checked swap range')
        relative = self.geometry(sample)
        if relative[:, 2].min() < self.floor or relative[:, 2].max() > 99:
            raise RuntimeError(f'Tool height guard at {np.round(relative[:, 2], 1).tolist()} mm')
        if relative[:, 0].min() < self.minimum_x or relative[:, 0].max() > 91 or relative[:, 1].min() < -50 or relative[:, 1].max() > 90:
            raise RuntimeError('Tool left the checked tabletop patch')
        return sample

    def photo(self, stage, fixed_target=None, wrist_target=(345, 394)):
        capture(self.directory, stage, self.observe)
        frames = {view: self.directory / f'{stage}_{view}.jpg' for view in ('fixed', 'wrist')}
        detector = find_orange_target if self.color == 'orange' else find_dark_parts
        found = {view: detector(path, view) for view, path in frames.items()}
        fixed_center = fixed_target or self.leg['source_px']
        fixed = nearby(found['fixed'], fixed_center, 28)
        fixed_is_partial = False
        if stage == 'test_lift' and self.color == 'orange' and fixed is None:
            fragments = find_orange_target(frames['fixed'], 'fixed', minimum_aspect=0.26)
            rising = [part for part in fragments if fixed_center[1] - 46 <= part['center_px'][1] <= fixed_center[1] - 8]
            grounded = nearby(fragments, fixed_center, 9)
            if grounded is None and len(rising) == 1:
                fixed = nearby(rising, fixed_center, 38)
                fixed_is_partial = fixed is not None
        sample = {
            'fixed': fixed,
            'fixed_is_partial': fixed_is_partial,
            'wrist': nearby(found['wrist'], wrist_target, 135),
            'fixed_candidates': found['fixed'],
        }
        self.report['vision'][stage] = sample
        return sample

    def check_box(self, readout, stage):
        wrist = readout['wrist']
        if wrist is None:
            raise RuntimeError(f'{self.color} was not identified in the wrist view at {stage}')
        x, y = wrist['center_px']
        x_bounds, y_bounds = (286, 410), (335, 437)
        if stage == 'hover_open':
            x_bounds, y_bounds = (235, 462), (212, 450)
        if stage == 'approach_open':
            x_bounds, y_bounds = (264, 437), (271, 451)
        if not x_bounds[0] <= x <= x_bounds[1] or not y_bounds[0] <= y <= y_bounds[1]:
            raise RuntimeError(f'{self.color} is off center at {stage}: {[x, y]}')
        if stage == 'pick_open':
            image = cv2.imread(str(self.directory / f'{stage}_wrist.jpg'))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if self.color == 'orange':
                mask = cv2.inRange(hsv, (4, 150, 140), (26, 255, 255))
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                mask = cv2.inRange(gray, 0, 91) & cv2.inRange(hsv[:, :, 1], 0, 121)
            box_x, _, width, _ = wrist['box_px']
            columns = np.flatnonzero((mask[380:405, box_x:box_x + width] > 0).sum(axis=0) >= 6) + box_x
            span = [int(columns[0]), int(columns[-1])] if len(columns) else None
            self.report['wrist_finger_overlap'] = span
            if len(columns) < 42 or columns[0] < 262 or columns[-1] > 462:
                raise RuntimeError(f'{self.color} does not fit between the fingers at the contact row: {span}')

    def require_destination_empty(self, stage):
        frame = cv2.imread(str(self.directory / f'{stage}_fixed.jpg'))
        x, y = self.leg['target_px']
        patch = frame[y - 10:y + 10, x - 10:x + 10]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        dark = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) < 112
        orange = cv2.inRange(hsv, (4, 150, 140), (26, 255, 255)) > 0
        if np.mean(dark | orange) > 0.13 or float(np.std(hsv[:, :, 2])) > 24:
            raise RuntimeError('The placement area is occupied or not visible enough')

    def verify_release(self, previous):
        target = self.leg['target_px']
        a = self.photo('placed_a', target)
        time.sleep(0.25)
        b = self.photo('placed_b', target)
        first = nearby(a['fixed_candidates'], target, 23)
        second = nearby(b['fixed_candidates'], target, 23)
        if first is None or second is None or np.linalg.norm(np.subtract(first['center_px'], second['center_px'])) > 2.5:
            raise RuntimeError('The placement was not stationary at the expected destination')
        other = nearby(b['fixed_candidates'], self.leg['source_px'], 17)
        if other is not None and other is not second:
            raise RuntimeError('The source still appears occupied by the moved color')
        self.report['placement'] = {'verified': True, 'target_px': target, 'observed_px': second['center_px'],
                                    'source_before_px': previous['fixed']['center_px'] if previous['fixed'] else None}

    def check_stack_base(self, stage, reference=None):
        frame = self.directory / f'{stage}_fixed.jpg'
        base = nearby(find_dark_parts(frame, 'fixed'), self.leg['target_px'], 8)
        if base is not None and reference is not None:
            _, _, width, height = base['box_px']
            _, _, original_width, original_height = reference['box_px']
            if height < original_height * 0.82 or not 0.8 <= width / original_width <= 1.25:
                base = None
        if base is None and reference is not None:
            base = match_support_edge(self.directory / 'clearance_fixed.jpg', frame, reference)
        if base is None:
            raise RuntimeError('Cannot verify the gray support at the planned spot')
        self.report.setdefault('stack_support', {})[stage] = base
        return base

    def verify_stack(self, base):
        x, y = base['center_px']
        candidate = []
        for suffix in ('a', 'b'):
            stage = f'stack_{suffix}'
            reading = self.photo(stage, (x, y - 11))
            block = nearby(reading['fixed_candidates'], (x, y - 11), 13)
            if block is None:
                raise RuntimeError('The orange block is absent above the gray support')
            cx, cy = block['center_px']
            frame = cv2.imread(str(self.directory / f'{stage}_fixed.jpg'))
            left, top, width, height = block['box_px']
            tail = frame[top + height:top + height + 8, int(x) - 7:int(x) + 8]
            gray = cv2.cvtColor(tail, cv2.COLOR_BGR2GRAY)
            if abs(cx - x) > 8 or not 6 <= y - cy <= 21 or np.mean(gray < 109) < 0.32:
                raise RuntimeError('The two blocks lack verified offset and visible support contact')
            candidate.append(block['center_px'])
            time.sleep(0.3)
        if np.linalg.norm(np.subtract(*candidate)) > 2.5:
            raise RuntimeError('The orange block is not stable on the support')
        self.report['placement'] = {'verified': True, 'observed_px': candidate[-1], 'support_px': [x, y]}

    def raise_clear(self, name, *, minimum_height=50, shoulder_only=False):
        """Raise the shoulder before an extended-arm traverse or closing empty fingers."""
        now = self.observe()
        for correction in range(4):
            height = self.geometry(now)[:, 2].min()
            if height >= minimum_height:
                return now
            base = tuple(now[axis] for axis in AXES)
            # Allow extra command margin for shoulder settling under load.
            offset = 1 if shoulder_only or correction < 2 or height < 30 else 2
            for count in range(96 if shoulder_only else 186, 25, -10):
                goal = base[:offset] + (base[offset] - count,) + base[offset + 1:]
                commanded = now | dict(zip(AXES, goal))
                try:
                    self.validate(commanded)
                except RuntimeError:
                    continue
                geometry = self.geometry(commanded)
                if geometry[:, 2].max() <= 93 and geometry[:, 2].min() >= height + 8:
                    now = self.go(goal, self.floor, f'{name}_{correction + 1}', corrections=False)
                    break
        if self.geometry(now)[:, 2].min() < minimum_height:
            raise RuntimeError('Measured clearance is insufficient for transverse movement')
        return now

    def run(self):
        leg = self.leg
        hover, approach, pick, source_high = leg['pickup']
        carry, settle, place = leg['destination']
        start = tuple(self.initial[axis] for axis in AXES)
        home = tuple(leg.get('park', start))
        above_home = home[:1] + (1920,) + home[2:]
        departure = start[:1] + (1875,) + start[2:]
        stack = leg.get('stack', False)
        at_source = jaw_open = at_destination = away_from_source = holding = released = high_enough = False
        torque_limit = read(self.bus, 'Torque_Limit', GRIPPER)
        with hold(self.bus, self.initial):
            try:
                rise = self.go(departure, 0, 'rise', corrections=False)
                if self.geometry(rise)[:, 2].min() < 39:
                    raise RuntimeError('Departure did not establish enough clearance for the traverse')
                self.go(RETURN_TRANSFER, 31, 'clearance')
                baseline = self.photo('clearance')
                if stack:
                    stack_reference = self.check_stack_base('clearance')
                elif not leg.get('correction'):
                    self.require_destination_empty('clearance')
                if baseline['fixed'] is None and leg.get('edge_pickup'):
                    self.go(EDGE_SOURCE_VIEW, 35, 'edge_source_view', corrections=False)
                    baseline = self.photo('edge_source_check')
                if baseline['fixed'] is None:
                    raise RuntimeError('The selected object was not identified near its expected source')
                high_enough = True
                if leg.get('edge_pickup') and hover[0] > 2112:
                    self.go(EDGE_APPROACH, 36, 'edge_approach', corrections=False)
                self.go(hover, 35, 'source_hover')
                at_source = True
                self.jaw(APERTURE)
                jaw_open = True
                self.check_box(self.photo('hover_open'), 'hover_open')
                self.go(approach, 14, 'source_approach')
                self.check_box(self.photo('approach_open'), 'approach_open')
                self.go(pick, 8, 'pick')
                before = self.photo('pick_open')
                self.check_box(before, 'pick_open')
                write(self.bus, 'Torque_Limit', GRIPPER, min(torque_limit, TORQUE_LIMIT))
                if read(self.bus, 'Torque_Limit', GRIPPER) != min(torque_limit, TORQUE_LIMIT):
                    raise RuntimeError('Could not verify reduced grip torque')
                self.report['jaw_at_close'] = self.jaw(max(2080, self.initial[GRIPPER]), unloaded=False)
                holding = True
                closed = self.photo('closed')
                self.go(hover, 8, 'test_lift')
                after = self.photo('test_lift')
                grasp_reference = before if before['fixed'] else baseline
                self.report['grasp_check'] = self.score(grasp_reference, closed, after)
                self.report['grasp_check']['fixed_lift_partly_occluded'] = after['fixed_is_partial']
                if self.report['grasp_check']['reward'] != 1 and 'proof_pose' in leg and closed['wrist'] and after['wrist']:
                    wrist_shift = np.linalg.norm(np.subtract(closed['wrist']['center_px'], after['wrist']['center_px']))
                    jaw_stall = self.report['jaw_at_close'] - self.initial[GRIPPER]
                    if wrist_shift < 12 and jaw_stall >= 24:
                        away_from_source = True
                        self.go(leg['proof_pose'], 38, 'lift_visibility_check')
                        proof = self.photo('lift_visibility_check', leg['proof_px'])
                        if nearby(proof['fixed_candidates'], leg['source_px'], 12) is not None:
                            raise RuntimeError('The source still appears occupied during the lift visibility check')
                        self.report['grasp_check'] = self.score(baseline, closed, proof)
                        self.report['grasp_check']['checked_at_visibility_pose'] = True
                if self.report['grasp_check']['reward'] != 1:
                    raise RuntimeError('The two cameras and gripper feedback did not verify a held object')
                if not away_from_source:
                    self.go(source_high, 33, 'source_carry_clearance')
                self.raise_clear('transport_clearance')
                if stack:
                    self.photo('stack_base_from_source', leg['target_px'])
                    try:
                        base = self.check_stack_base('stack_base_from_source', stack_reference)
                    except RuntimeError:
                        away_from_source = True
                        self.go(CORRECTION_VIEW, 38, 'clearance_after_pick')
                        self.photo('clearance_after_pick', leg['target_px'])
                        base = self.check_stack_base('clearance_after_pick', stack_reference)
                    if not away_from_source:
                        away_from_source = True
                        self.go(RETURN_TRANSFER, 38, 'transport_midpoint')
                elif leg.get('correction'):
                    away_from_source = True
                    self.go(CORRECTION_VIEW, 38, 'clearance_after_pick')
                    self.photo('clearance_after_pick', leg['target_px'])
                    self.require_destination_empty('clearance_after_pick')
                else:
                    away_from_source = True
                    self.go(RETURN_TRANSFER, 38, 'transport_midpoint')
                at_destination = True
                self.go(carry, 38, 'carry')
                self.go(settle, 37 if stack else 24, 'pre_place')
                if stack:
                    self.photo('stack_approach', leg['target_px'])
                    base = self.check_stack_base('stack_approach', stack_reference)
                    normal_step_counts = self.step_counts
                    self.step_counts = 24
                    try:
                        self.go(place, 38, 'place', corrections=False)
                    finally:
                        self.step_counts = normal_step_counts
                else:
                    self.go(place, 8, 'place')
                before_release = self.photo('before_release', leg['target_px'])
                if stack:
                    base = self.check_stack_base('before_release', stack_reference)
                    observed = before_release['fixed']
                    if not observed or np.linalg.norm(np.subtract(
                            observed['center_px'], (base['center_px'][0], base['center_px'][1] - 14))) > 10:
                        raise RuntimeError('The held cube did not align with the support before release')
                    starting_aperture = self.observe()[GRIPPER]
                    self.jaw(min(APERTURE, starting_aperture + 24))
                    released = True
                    holding = False
                    self.report['released_at_destination'] = True
                    for increment in (56, 96):
                        time.sleep(0.22)
                        self.jaw(min(APERTURE, starting_aperture + increment))
                    self.photo('stack_release_open', leg['target_px'])
                self.jaw(APERTURE)
                released = True
                self.report['released_at_destination'] = True
                holding = False
                if stack:
                    time.sleep(0.85)
                    self.photo('stack_fingers_open', leg['target_px'])
                    away, back, high = leg['detach']
                    self.go(away, 38, 'stack_clear_fixed_jaw')
                    self.photo('stack_jaw_clear', leg['target_px'])
                    self.go(back, 38, 'stack_retreat_from_support')
                    self.photo('stack_back_clear', leg['target_px'])
                    self.go(high, 43, 'stack_raise_away_from_support')
                    self.raise_clear('stack_vertical_retreat', minimum_height=58, shoulder_only=True)
                    self.photo('stack_raise_clear', leg['target_px'])
                    retreat = self.observe()
                else:
                    retreat = self.go(carry, 8, 'release_retreat')
                if self.geometry(retreat)[:, 2].min() < 36:
                    raise RuntimeError('The post-release retreat did not establish enough clearance')
                self.raise_clear('clear_of_placed_object')
                self.go(CORRECTION_VIEW if stack else RETURN_TRANSFER, 45 if stack else 34, 'placement_inspection')
                self.jaw(IDLE_JAW)
                jaw_open = False
                at_destination = False
                at_source = False
                if stack:
                    try:
                        self.verify_stack(base)
                    except RuntimeError:
                        self.go(RETURN_TRANSFER, 33, 'stack_alternate_inspection')
                        self.verify_stack(base)
                else:
                    self.verify_release(baseline)
            except Exception as exc:
                self.report['phase_error'] = str(exc)
                raise
            finally:
                try:
                    if holding:
                        if at_destination:
                            self.go(carry, 8, 'rollback_raise')
                            self.raise_clear('rollback_height')
                        if at_destination or away_from_source:
                            self.go(source_high, 34, 'rollback_carry')
                            at_destination = False
                        if leg.get('edge_pickup'):
                            self.go(hover, 34, 'rollback_hover')
                            self.go(approach, 14, 'rollback_approach')
                        self.go(pick, 8, 'return_to_source', corrections=False)
                        self.jaw(APERTURE)
                    if jaw_open and stack and released:
                        actual_z = float(self.geometry(positions(self.bus, self.initial))[:, 2].min())
                        if actual_z < 35:
                            raise RuntimeError('The released stack is too close for a clear retreat')
                        self.floor = min(38, actual_z - 0.5)
                        self.raise_clear('stack_recovery_raise', minimum_height=58, shoulder_only=True)
                    elif jaw_open:
                        actual_z = float(self.geometry(positions(self.bus, self.initial))[:, 2].min())
                        if 0 <= actual_z < 9:
                            self.floor = actual_z - 0.7
                            self.raise_clear('raise_before_retreat')
                        retreat = carry if at_destination else hover
                        if actual_z >= 9:
                            self.go(retreat, 8, 'jaw_clearance', corrections=False)
                        self.raise_clear('clear_of_returned_object')
                    if high_enough or at_source or at_destination:
                        self.go(RETURN_TRANSFER, 45 if stack and released else 29, 'return_transfer')
                    self.go(above_home, 26, 'return_above_home', corrections=False)
                    if jaw_open:
                        self.jaw(IDLE_JAW)
                    self.go(home, 0, 'return_home', corrections=False)
                    self.photo('reset', leg['target_px'] if released else leg['source_px'])
                finally:
                    write(self.bus, 'Torque_Limit', GRIPPER, torque_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('plan', type=str)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run_id.isascii() or not args.run_id.replace('_', '').isalnum():
        parser.error('Invalid run ID')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    from settings import require_hardware_enabled
    require_hardware_enabled()
    load = lambda name: json.loads((CHECKS / name).read_text(encoding='utf-8'))
    leg = load(args.plan)
    if load('swap_supervision.json') != (STACK_APPROVAL if leg.get('stack') else SWAP_APPROVAL):
        raise RuntimeError('Vision-language approval for both objects is missing')
    directory = CHECKS / args.run_id
    if args.run:
        directory.mkdir(mode=0o700, exist_ok=False)
    anchor = load('base_alignment_p4_verified.json')['raw_positions']
    models = [load('base_alignment_p4_checked.json')['encoder_zeros'], load('base_alignment_checked.json')['encoder_zeros']]
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id='so101-workbench', calibration_dir=LAB / 'calibration'))
    bus = robot.bus
    report = {'plan': leg, 'motion_requested': args.run, 'poses': [], 'vision': {}, 'planned_aperture': APERTURE}
    bus.connect()
    try:
        initial = report['initial'] = positions(bus, bus.motors)
        if not robot.is_calibrated or any(abs(value - leg['start'][joint]) > 3 for joint, value in initial.items()):
            raise RuntimeError('The robot state differs from the checked start')
        for joint in initial:
            if read(bus, 'Torque_Enable', joint) or read(bus, 'Operating_Mode', joint):
                raise RuntimeError(f'{joint} requires torque off and position mode')
        models.append({joint: (bounds.range_min + bounds.range_max) / 2 for joint, bounds in robot.calibration.items()})
        task = SwapTransfer(bus, initial, anchor, models, directory, report, leg=leg)
        report['waypoint_predictions_mm'] = []
        waypoints = [RETURN_TRANSFER, *leg['pickup'], *leg['destination'], *leg.get('detach', ())]
        if leg.get('stack') and len(leg['detach']) != 3:
            raise RuntimeError('A checked fixed-jaw retreat is required')
        if leg.get('edge_pickup'):
            waypoints.extend((EDGE_APPROACH, EDGE_SOURCE_VIEW))
        if leg.get('correction') or leg.get('stack'):
            waypoints.append(CORRECTION_VIEW)
        if 'proof_pose' in leg:
            waypoints.append(leg['proof_pose'])
        if 'park' in leg:
            waypoints.append(leg['park'])
        for waypoint in waypoints:
            pose = task.validate(initial | dict(zip(AXES, waypoint)))
            for axis in AXES:
                bounds = robot.calibration[axis]
                if not bounds.range_min + 24 <= pose[axis] <= bounds.range_max - 24:
                    raise RuntimeError(f'{axis} waypoint does not leave enough joint margin')
            report['waypoint_predictions_mm'].append(task.geometry(pose).round(2).tolist())
        if args.run:
            task.run()
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        try:
            report['final_torque'] = {joint: read(bus, 'Torque_Enable', joint) for joint in bus.motors}
            report['released_positions'] = positions(bus, bus.motors)
            if any(report['final_torque'].values()):
                raise RuntimeError('Unexpected active torque after the swap leg')
        finally:
            bus.disconnect(disable_torque=False)
            content = json.dumps(report, indent=2) + '\n'
            if args.run:
                (directory / 'report.json').write_text(content, encoding='utf-8')
            print(content, end='')


if __name__ == '__main__':
    main()
