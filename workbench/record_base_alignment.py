"""Record a stationary, hand-placed SO-101 fingertip alignment sample."""

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import struct
from xml.etree import ElementTree

import cv2
import numpy as np


def vector(node, name, default='0 0 0'):
    return np.fromstring(node.get(name, default) if node is not None else default, sep=' ')


def rotation(axis, angle):
    matrix, _ = cv2.Rodrigues(np.asarray(axis, dtype=float) * angle)
    return matrix


def origin(node):
    roll, pitch, yaw = vector(node, 'rpy')
    transform = np.eye(4)
    transform[:3, :3] = rotation((0, 0, 1), yaw) @ rotation((0, 1, 0), pitch) @ rotation((1, 0, 0), roll)
    transform[:3, 3] = vector(node, 'xyz')
    return transform


def fixed_tip_in_gripper(model, root):
    for visual in root.findall('./link/visual'):
        mesh = visual.find('./geometry/mesh')
        if mesh is not None and Path(mesh.get('filename')).name == 'wrist_roll_follower_so101_v1.stl':
            raw = (model.parent / mesh.get('filename')).read_bytes()
            triangles = struct.unpack_from('<I', raw, 80)[0]
            if len(raw) != 84 + 50 * triangles:
                raise ValueError('Unexpected fingertip STL format')
            dtype = np.dtype([('normal', '<f4', 3), ('points', '<f4', (3, 3)), ('flags', '<u2')])
            vertices = np.frombuffer(raw, dtype=dtype, count=triangles, offset=84)['points'].reshape(-1, 3)
            visual_frame = origin(visual.find('origin'))
            points = vertices * vector(mesh, 'scale', '1 1 1') @ visual_frame[:3, :3].T + visual_frame[:3, 3]
            tip_surface = np.unique(points[points[:, 2] <= points[:, 2].min() + 0.0005], axis=0)
            return (tip_surface.min(axis=0) + tip_surface.max(axis=0)) * 0.5
    raise ValueError('Could not find the fixed-jaw mesh in this model')


@lru_cache(maxsize=4)
def model_geometry(model):
    root = ElementTree.parse(model).getroot()
    joints = {joint.find('child').get('link'): joint for joint in root.findall('joint')}
    chain = []
    child = root.find("./joint[@name='gripper']/parent").get('link')
    while child in joints:
        joint = joints[child]
        chain.append(joint)
        child = joint.find('parent').get('link')
    frames = tuple(
        (origin(joint.find('origin')), joint.get('name') if joint.get('type') != 'fixed' else None,
         vector(joint.find('axis'), 'xyz'))
        for joint in reversed(chain)
    )
    return frames, np.append(fixed_tip_in_gripper(model, root), 1.0)


def fingertip_mm(model, positions, encoder_zero):
    frames, fixed_tip = model_geometry(model)
    position = np.eye(4)
    for frame, name, axis in frames:
        position = position @ frame
        if name is not None:
            angle = (positions[name] - encoder_zero[name]) * math.tau / 4095
            turn = np.eye(4)
            turn[:3, :3] = rotation(axis, angle)
            position = position @ turn
    return (position @ fixed_tip)[:3] * 1000


def fingertip(model, positions, encoder_zero):
    return [round(float(coord), 2) for coord in fingertip_mm(model, positions, encoder_zero)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plane', type=Path, required=True)
    parser.add_argument('--reading', type=Path, required=True)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--arm-calibration', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--px', type=float, nargs=2, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plane = json.loads(args.plane.read_text(encoding='utf-8'))
    if plane['inferred_corner'] is not None:
        raise ValueError('The paper plane needs four observed corners')
    reading = json.loads(args.reading.read_text(encoding='utf-8'))
    if len(reading['servos']) != 6 or any(not entry['ok'] or entry['torque_enabled'] is not False for entry in reading['servos']):
        raise ValueError('Expected six responding servos with torque verified off')
    positions = {entry['name']: entry['raw_position'] for entry in reading['servos']}
    calibration = json.loads(args.arm_calibration.read_text(encoding='utf-8'))
    angle_zeros = {
        'lerobot_range_midpoint': {name: (entry['range_min'] + entry['range_max']) / 2 for name, entry in calibration.items()},
        'encoder_half_turn': dict.fromkeys(positions, 2047.5),
    }
    paper = cv2.perspectiveTransform(np.array([[args.px]], dtype=np.float64), np.array(plane['pixel_to_paper_mm']))[0, 0]
    result = {
        'label': args.label,
        'feature_px': args.px,
        'paper_mm': [round(float(coord), 2) for coord in paper],
        'raw_positions': positions,
        'fk_candidates_mm': {
            f'{model.stem}/{convention}': fingertip(model, positions, zeros)
            for model in args.model for convention, zeros in angle_zeros.items()
        },
    }
    data = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(data)
    print(data, end='')


if __name__ == '__main__':
    main()
