"""Run a guarded real-arm grasp episode with camera-based scoring and reset."""

import argparse
import json
import math
import signal
import time

import cv2
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from hover_probe import CHECKS, LAB, LIFT_JOINT, PORT, positions, read, stop, write
from learn_lift import capture
from orange_approach_probe import GRIPPER, hold
from record_base_alignment import fingertip_mm
from supervise_snapshot import find_orange_target


AXES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex')
TRANSFER = (2051, 1888, 2638, 1581)
RETURN_TRANSFER = (2051, 1858, 2638, 1581)
HOVER = (1995, 1898, 2570, 1674)
CHECK = (1995, 2035, 2564, 1626)
PICK = (1995, 2054, 2578, 1594)
FORWARD_RETRY = ((2005, 1975, 2520, 1639), (2005, 2109, 2437, 1679), (2005, 2143, 2468, 1614))
WIDE_RETRY = ((1982, 2017, 2459, 1658), (1982, 2146, 2369, 1710), (1982, 2183, 2408, 1634))
OPEN = 2370
WIDE_OPEN = 2460


class GraspEpisode:
    step_counts = 65

    def __init__(self, bus, initial, anchor, models, directory, report):
        self.bus = bus
        self.initial = initial
        self.directory = directory
        self.report = report
        self.model = LAB / 'geometry' / 'so101_new_calib.urdf'
        self.references = [(zeros, fingertip_mm(self.model, anchor, zeros)) for zeros in models]
        self.floor = 0
        self.aperture = report.get('planned_aperture', OPEN)

    def geometry(self, sample):
        return np.array([fingertip_mm(self.model, sample, zeros) - reference for zeros, reference in self.references])

    def validate(self, sample):
        limits = ((1950, 2168), (1825, 2208), (2320, 2720), (1400, 1745))
        for name, (lower, upper) in zip(AXES, limits):
            if not lower <= sample[name] <= upper:
                raise RuntimeError(f'{name} outside the approved joint window: {sample[name]}')
        if abs(sample['wrist_roll'] - self.initial['wrist_roll']) > 10 or not 2060 <= sample[GRIPPER] <= self.aperture + 8:
            raise RuntimeError('Wrist roll or jaw aperture exceeded the allowed window')
        relative = self.geometry(sample)
        if relative[:, 2].min() < self.floor or relative[:, 2].max() > 90:
            raise RuntimeError(f'Estimated tool height left the current workspace: {np.round(relative[:, 2], 1)}')
        if np.any(np.abs(relative[:, 0]) > 48) or relative[:, 1].min() < -12 or relative[:, 1].max() > 72:
            raise RuntimeError('Estimated tool position left the validated tabletop patch')
        return sample

    def observe(self):
        sample = self.validate(positions(self.bus, self.initial))
        if any(read(self.bus, 'Status', joint) for joint in (*AXES, GRIPPER)):
            raise RuntimeError('A servo reported a fault')
        return sample

    def settle(self):
        deadline = time.monotonic() + 4.0
        last = self.observe()
        since = time.monotonic()
        while time.monotonic() < deadline:
            now = self.observe()
            if any(abs(now[axis] - last[axis]) > 1 for axis in AXES):
                since = time.monotonic()
                last = now
            if time.monotonic() - since > 0.44:
                return now
            time.sleep(0.035)
        raise RuntimeError('Arm did not settle under feedback')

    def go(self, target, floor, name, corrections=True):
        self.floor = floor
        now = self.observe()
        steps = max(1, math.ceil(max(abs(value - now[axis]) for axis, value in zip(AXES, target)) / self.step_counts))
        ramp = np.linspace([now[axis] for axis in AXES], target, steps + 1)[1:]
        for waypoint in ramp:
            desired = dict(zip(AXES, np.rint(waypoint).astype(int).tolist()))
            self.validate(now | desired)
            for axis, value in desired.items():
                write(self.bus, 'Goal_Position', axis, value)
            now = self.settle()
        difference = [goal - now[axis] for axis, goal in zip(AXES, target)]
        if corrections and max(map(abs, difference)) > 10:
            adjusted = {axis: goal + max(-21, min(21, delta)) for axis, goal, delta in zip(AXES, target, difference)}
            try:
                self.validate(now | adjusted)
            except RuntimeError:
                pass
            else:
                for axis, value in adjusted.items():
                    write(self.bus, 'Goal_Position', axis, value)
                now = self.settle()
        self.report['poses'].append({'name': name, 'positions': now, 'estimated_mm': np.round(self.geometry(now), 1).tolist()})
        return now

    def photo(self, stage):
        capture(self.directory, stage, self.observe)
        found = {}
        for view in ('fixed', 'wrist'):
            targets = find_orange_target(self.directory / f'{stage}_{view}.jpg', view)
            found[view] = targets[0] if len(targets) == 1 else None
        self.report['vision'][stage] = found
        return found

    def jaw(self, goal, unloaded=True):
        write(self.bus, 'Goal_Position', GRIPPER, goal)
        if read(self.bus, 'Goal_Position', GRIPPER) != goal:
            raise RuntimeError('Gripper target was not received')
        deadline = time.monotonic() + 6.0
        recent = self.observe()[GRIPPER]
        since = time.monotonic()
        while time.monotonic() < deadline:
            actual = self.observe()[GRIPPER]
            if abs(actual - goal) <= 9:
                return actual
            if abs(actual - recent) > 2:
                recent = actual
                since = time.monotonic()
            if not unloaded and time.monotonic() - since > 0.5:
                return actual
            time.sleep(0.05)
        if unloaded:
            raise RuntimeError('Unloaded gripper did not reach its target')
        return actual

    def align(self, stage, x_range, y_range):
        found = self.photo(stage)
        wrist = found['wrist']
        if wrist is None:
            raise RuntimeError('The orange block was not located in the wrist view')
        x, y = wrist['center_px']
        if not x_range[0] <= x <= x_range[1] or not y_range[0] <= y <= y_range[1]:
            raise RuntimeError(f'Orange block is outside the jaw alignment window at {stage}: {[x, y]}')
        return found

    def verify_finger_overlap(self):
        image = cv2.imread(str(self.directory / 'pick_open_wrist.jpg'))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (4, 150, 140), (26, 255, 255))
        columns = np.flatnonzero((mask[380:405, :] > 0).sum(axis=0) >= 6)
        contact = [int(columns[0]), int(columns[-1])] if len(columns) else None
        self.report['wrist_finger_overlap'] = {'row_band': [380, 405], 'orange_span_px': contact}
        if len(columns) < 45 or columns[0] < 262 or columns[-1] > 462:
            raise RuntimeError('The block did not fit between the open fingertips at the contact row')

    def score(self, before, closed, after):
        center = lambda pair, view: np.array(pair[view]['center_px']) if pair[view] else None
        fixed_before, fixed_after = center(before, 'fixed'), center(after, 'fixed')
        wrist_before, wrist_after = center(closed, 'wrist'), center(after, 'wrist')
        if any(value is None for value in (fixed_before, fixed_after, wrist_before, wrist_after)):
            return {'reward': 0, 'reason': 'orange_not_visible_in_both_views'}
        fixed_shift = float(np.linalg.norm(fixed_after - fixed_before))
        wrist_shift = float(np.linalg.norm(wrist_after - wrist_before))
        jaw_stall = self.report['jaw_at_close'] - max(2080, self.initial[GRIPPER])
        held = fixed_after[1] < fixed_before[1] - 5 and wrist_shift <= 19 and jaw_stall >= 24
        return {'reward': int(held), 'fixed_shift_px': round(fixed_shift, 2), 'wrist_shift_px': round(wrist_shift, 2)}

    def run(self, home_pose, recovery, route):
        hover, check, pick = route
        home = tuple(home_pose[axis] for axis in AXES)
        start = tuple(self.initial[axis] for axis in AXES)
        raised = start[:1] + ((1840 if recovery else 1920),) + start[2:]
        high = home[:1] + (1920,) + home[2:]
        grip_depth_reached = False
        transit_started = False
        hover_reached = False
        jaw_open = False
        torque_limit = read(self.bus, 'Torque_Limit', GRIPPER)
        with hold(self.bus, self.initial):
            try:
                self.photo('start')
                self.go(raised, 0, 'rise', corrections=False)
                clearance = self.geometry(self.observe())[:, 2].min()
                if clearance < 33:
                    raise RuntimeError(f'Rise did not create enough clearance: {clearance:.1f} mm')
                transit_started = True
                transfer = RETURN_TRANSFER if self.report.get('raised_transfer') else TRANSFER
                self.go(transfer, 32, 'transfer')
                self.go(hover, 35, 'hover')
                hover_reached = True
                jaw_open = True
                self.jaw(self.aperture)
                self.align('hover_open', (280, 437), (220, 450))
                self.go(check, 14, 'approach')
                self.align('approach_open', (293, 399), (282, 444))
                self.go(pick, 11, 'pick')
                before = self.align('pick_open', (294, 390), (315 if route == (HOVER, CHECK, PICK) else 345, 434))
                if self.aperture == WIDE_OPEN:
                    self.verify_finger_overlap()
                grip_depth_reached = True
                write(self.bus, 'Torque_Limit', GRIPPER, min(torque_limit, 160))
                if read(self.bus, 'Torque_Limit', GRIPPER) != min(torque_limit, 160):
                    raise RuntimeError('Could not verify the reduced grip torque limit')
                self.report['jaw_at_close'] = self.jaw(max(2080, self.initial[GRIPPER]), unloaded=False)
                closed = self.photo('closed')
                self.go(hover, 11, 'test_lift')
                after = self.photo('test_lift')
                self.report['outcome'] = self.score(before, closed, after)
            finally:
                try:
                    if grip_depth_reached:
                        self.go(pick, 10, 'lower_for_release', corrections=False)
                        self.jaw(self.aperture)
                    if transit_started:
                        if hover_reached:
                            self.go(hover, 10, 'retreat', corrections=False)
                        if jaw_open:
                            self.jaw(self.initial[GRIPPER])
                        self.go(RETURN_TRANSFER, 30, 'reverse_transfer')
                        self.go(high, 30, 'back_above_start', corrections=False)
                    if transit_started or not recovery:
                        self.go(home, 0, 'reset', corrections=False)
                    self.photo('reset')
                finally:
                    write(self.bus, 'Torque_Limit', GRIPPER, torque_limit)
