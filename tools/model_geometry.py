"""Static F1 URDF/STL calculations; no simulator or parameter estimation."""
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, :3] = rotation([float(v) for v in element.get('rpy', '0 0 0').split()])
        transform[:3, 3] = [float(v) for v in element.get('xyz', '0 0 0').split()]
    return transform


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)
    return transform


def forward_kinematics(robot, angles):
    joints = robot.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    roots = {l.get('name') for l in robot.findall('link')} - children
    if len(roots) != 1:
        raise ValueError('URDF must have one root')
    transforms = {roots.pop(): np.eye(4)}
    remaining = list(joints)
    while remaining:
        progress = False
        for joint in remaining[:]:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            if parent not in transforms:
                continue
            transform = transforms[parent] @ origin(joint.find('origin'))
            if joint.get('type') != 'fixed':
                transform = transform @ axis_rotation([float(v) for v in joint.find('axis').get('xyz').split()], angles[joint.get('name')])
            transforms[child] = transform
            remaining.remove(joint)
            progress = True
        if not progress:
            raise ValueError('Disconnected or cyclic URDF')
    return transforms


def mesh_vertices(path):
    data = Path(path).read_bytes()
    if data.startswith(b'version https://git-lfs'):
        raise ValueError(f'LFS pointer instead of mesh: {path}')
    if len(data) >= 84:
        count = struct.unpack_from('<I', data, 80)[0]
        if len(data) == 84 + 50*count:
            dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
            return np.frombuffer(data, dtype=dtype, count=count, offset=84)['vertices'].reshape(-1, 3).astype(float)
    vertices = [[float(v) for v in line.split()[1:]] for line in data.decode('ascii').splitlines() if line.strip().startswith('vertex ')]
    if not vertices:
        raise ValueError(f'No STL vertices: {path}')
    return np.asarray(vertices)


def collision_vertices(urdf, link):
    points = []
    for collision in link.findall('collision'):
        mesh = collision.find('geometry/mesh')
        if mesh is None:
            raise ValueError('This audit expects mesh collision geometry')
        path = Path(urdf).parent / mesh.get('filename')
        vertices = mesh_vertices(path)
        vertices *= np.array([float(v) for v in mesh.get('scale', '1 1 1').split()])
        transform = origin(collision.find('origin'))
        points.append(vertices @ transform[:3, :3].T + transform[:3, 3])
    return np.concatenate(points)


