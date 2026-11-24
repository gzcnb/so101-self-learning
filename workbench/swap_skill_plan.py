"""Position a new transfer relative to physically verified workbench grasps."""

import json
from math import acos, degrees, tau
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from record_base_alignment import model_geometry, rotation
from swap_vision import pixel_to_table
from settings import PROFILE_DIR


ROOT = PROFILE_DIR
AXES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex')
PARK = (2112, 2025, 2673, 1460)
GRAY_CUBE_HEIGHT_MM = 28
STACK_RELEASE_CLEARANCE_MM = 8
MIN_STACK_RELEASE_CLEARANCE_MM = 4


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


class WorkbenchPlanner:
    def __init__(self, root=ROOT):
        self.h = np.array(read_json(root / 'paper_plane.json')['pixel_to_paper_mm'])
        table = read_json(root / 'base_alignment_p4_checked.json')
        self.basis = np.array(table['paper_to_base_xy_mm'])[:, :2]
        self.zeros = table['encoder_zeros']
        self.frames, self.tip = model_geometry(Path(__file__).resolve().parent / 'geometry/so101_new_calib.urdf')
        self.grasps = []
        self.placements = []
        for folder in ('swap_orange_temp_01', 'swap_gray_orange_01', 'swap_orange_gray_05'):
            report = read_json(root / folder / 'report.json')
            if report['grasp_check']['reward'] != 1:
                raise ValueError(f'Unverified workbench reference {folder}')
            source = report['vision']['clearance']['fixed']['center_px']
            color = report['plan']['color']
            poses = {item['name']: item['positions'] for item in report['poses']}
            self.grasps.append((source, poses['pick'], color))
            if (report.get('placement') or {}).get('verified'):
                self.placements.append((report['placement']['observed_px'], poses['place'], color))
            elif folder == 'swap_orange_temp_01':
                verified = read_json(root / 'swap_stage1_verified.json')
                if verified['placement_verified']:
                    self.placements.append((verified['fixed_centers_px'][-1], poses['place'], color))

    def pose(self, sample):
        transform = np.eye(4)
        for origin, joint, axis in self.frames:
            transform = transform @ origin
            if joint is not None:
                angle = (sample[joint] - self.zeros[joint]) * tau / 4095
                transform[:3, :3] = transform[:3, :3] @ rotation(axis, angle)
        return (transform @ self.tip)[:3] * 1000, transform[:3, :3]

    def table_distance(self, a, b):
        return float(np.linalg.norm(pixel_to_table(a, self.h) - pixel_to_table(b, self.h)))

    def route(self, references, color, point, bias, stages):
        pixel, nominal, _ = min(references, key=lambda ref: self.table_distance(point, ref[0]) + (6 if ref[2] != color else 0))
        if self.table_distance(point, pixel) > 78:
            raise ValueError('落点超出实测抓放位置周围的可用范围')
        anchor, orientation = self.pose(nominal)
        offset = self.basis @ (pixel_to_table(point, self.h) - pixel_to_table(pixel, self.h)) + bias
        seed = [nominal[joint] for joint in AXES]
        bounds = ([1895, 1810, 1910, 1378], [2223, 2485, 2790, 1906])
        commands = []
        for elevation, tilt in stages:
            target = anchor + np.r_[offset, elevation]

            def residual(values):
                position, wrist = self.pose(nominal | dict(zip(AXES, values)))
                position[:2] += 16 * (wrist[:2, 0] - orientation[:2, 0])
                return np.r_[position - target, 0.65 * (degrees(acos(np.clip(wrist[2, 2], -1, 1))) - tilt)]

            fit = least_squares(residual, seed, bounds=bounds)
            seed = np.rint(fit.x).astype(int).tolist()
            if not fit.success or np.linalg.norm(residual(seed)) > 2.5:
                raise ValueError('当前落点超出机械臂验证过的行程')
            commands.append(seed)
        return commands

    def transfer(self, color, source, target, encoders, *, bias=(0, 0), extra_lift=0,
                 correction=False, stack=False, placement_trim=None):
        if self.table_distance(source, target) < (24 if correction else 34):
            raise ValueError('抓取点和放置点距离不足')
        pickup_bias = np.array(bias)
        edge_pickup = False
        try:
            pickup = self.route(self.grasps, color, source, pickup_bias, [(44, 35), (25, 28), (2, 27), (56, 35)])
        except ValueError:
            if not stack:
                raise
            lateral = 5 if source[0] < 290 else -9
            pickup_bias = np.clip(pickup_bias + [5, lateral], -15, 15)
            pickup = self.route(self.grasps, color, source, pickup_bias, [(44, 32), (25, 28), (2, 26), (55, 32)])
            edge_pickup = True
        placement_trim = placement_trim or {'xy_mm': (0, 0), 'release_mm': STACK_RELEASE_CLEARANCE_MM}
        placement_bias = np.array(placement_trim['xy_mm'])
        if stack:
            release = GRAY_CUBE_HEIGHT_MM + max(MIN_STACK_RELEASE_CLEARANCE_MM, placement_trim['release_mm'])
            tilt = 24
            try:
                destination = self.route(self.placements, color, target, placement_bias,
                                         [(64, 35), (GRAY_CUBE_HEIGHT_MM + 21, tilt), (release, tilt)])
            except ValueError:
                tilt = 26
                destination = self.route(self.placements, color, target, placement_bias,
                                         [(64, 35), (GRAY_CUBE_HEIGHT_MM + 21, tilt), (release, tilt)])
            pan = (self.zeros['shoulder_pan'] - destination[-1][0]) * tau / 4095
            fixed_jaw_offset = 5.5 * np.array([-np.sin(pan), np.cos(pan)])
            forward = np.array([np.cos(pan), np.sin(pan)])
            separation = placement_bias + fixed_jaw_offset
            detach = self.route(self.placements, color, target, separation - 3 * forward,
                                [(release + 2, tilt)])
            detach += self.route(self.placements, color, target, separation - 15 * forward,
                                 [(release + 5, tilt), (62, 27)])
        else:
            destination = self.route(self.placements, color, target, placement_bias,
                                     [(56, 35), (35 + extra_lift, 29), (1, 27)])
            detach = []
        return {'color': color, 'source_px': np.rint(source).astype(int).tolist(),
                'target_px': np.rint(target).astype(int).tolist(), 'start': encoders,
                'pickup': pickup, 'pickup_bias_mm': pickup_bias.tolist(), 'destination': destination,
                'park': PARK, 'correction': correction, 'stack': stack, 'edge_pickup': edge_pickup, 'detach': detach,
                'proof_pose': [1962, 1989, 2459, 1658], 'proof_px': [318, 270]}

    def stack_feedback(self, report, support, previous=None, *, lower=True):
        previous = previous or {'xy_mm': (0, 0), 'release_mm': STACK_RELEASE_CLEARANCE_MM}
        reading = (report['vision'].get('before_release') or {}).get('fixed')
        if reading is None:
            raise ValueError('松爪前的定位证据不足')
        center = reading['center_px']
        # Allow for the image movement caused by lowering the held cube.
        descent = max(0, min(previous['release_mm'] - MIN_STACK_RELEASE_CLEARANCE_MM, 2)) if lower else 0
        expected = (support[0], support[1] - 11 - previous['release_mm'] * 0.38)
        delta = self.basis @ (pixel_to_table(expected, self.h) - pixel_to_table(center, self.h))
        trim = np.clip(np.asarray(previous['xy_mm']) + np.clip(delta, -5, 5), -9, 9)
        return {'target_px': list(support), 'xy_mm': trim.round(2).tolist(),
                'release_mm': max(MIN_STACK_RELEASE_CLEARANCE_MM, previous['release_mm'] - descent)}

    def pickup_feedback(self, report, bias):
        sample = next((report['vision'][key]['wrist'] for key in ('pick_open', 'approach_open', 'hover_open')
                       if report['vision'].get(key, {}).get('wrist')), None)
        if sample is None:
            return tuple(bias)
        x, y = sample['center_px']
        span = report.get('wrist_finger_overlap')
        lateral = (347 - x) / 4.1
        if span and span[0] < 264:
            lateral = max(lateral, (272 - span[0]) / 4.1)
        if span and span[1] > 457:
            lateral = min(lateral, (447 - span[1]) / 4.1)
        longitudinal = (392 - y) / 5.5
        joint = next((pose['positions']['shoulder_pan'] for pose in reversed(report['poses']) if pose['name'] == 'pick'), 2060)
        angle = (self.zeros['shoulder_pan'] - joint) * tau / 4095
        forward = np.array([np.cos(angle), np.sin(angle)])
        sideways = np.array([-np.sin(angle), np.cos(angle)])
        adjustment = np.clip(longitudinal, -8, 8) * forward + np.clip(lateral, -9, 9) * sideways
        return np.clip(np.array(bias) + adjustment, -20, 20).tolist()
