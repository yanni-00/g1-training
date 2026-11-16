"""Batched F1 URDF forward/inverse kinematics. No simulator or SciPy dependency.

The IK is pose-level processing, not evidence of dynamic feasibility/contact.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from model_geometry import origin, collision_vertices

SIDES = ("left", "right")
LEG_SUFFIXES = ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")


def skew(v):
    result = np.zeros((*v.shape[:-1], 3, 3), dtype=float)
    result[..., 0, 1], result[..., 0, 2] = -v[..., 2], v[..., 1]
    result[..., 1, 0], result[..., 1, 2] = v[..., 2], -v[..., 0]
    result[..., 2, 0], result[..., 2, 1] = -v[..., 1], v[..., 0]
    return result


def rot_axis(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = skew(axis)
    a = np.asarray(angle)[..., None, None]
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def rot_z(angle):
    return rot_axis(np.array([0., 0., 1.]), angle)


def quat_rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=-1) == 0):
        raise ValueError("Finite nonzero wxyz quaternions required")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(*q.shape[:-1], 3, 3)


def apply(r, v):
    return np.einsum("...ij,...j->...i", r, v)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def orientation_error(current, target):
    # Spatial SO(3) error, safe for the modest retargeted foot corrections here.
    return .5 * np.cross(np.swapaxes(current, -1, -2), np.swapaxes(target, -1, -2)).sum(axis=-2)


def orientation_angle(current, target):
    relative = target @ np.swapaxes(current, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)
    return np.arccos(cosine)


class F1Kinematics:
    def __init__(self, urdf):
        self.urdf = Path(urdf)
        self.robot = ET.parse(self.urdf).getroot()
        self.joints = [j for j in self.robot.findall("joint") if j.get("type") != "fixed"]
        self.names = [j.get("name") for j in self.joints]
        self.by_name = {j.get("name"): j for j in self.robot.findall("joint")}
        self.legs, self.vertices, self.centers, self.lower, self.upper = {}, {}, {}, {}, {}
        for side in SIDES:
            chain = []
            for suffix in LEG_SUFFIXES:
                name = f"{side}_{suffix}_joint"
                joint = self.by_name[name]
                chain.append({"name": name, "parent": joint.find("parent").get("link"),
                              "child": joint.find("child").get("link"), "origin": origin(joint.find("origin")),
                              "axis": np.array([float(v) for v in joint.find("axis").get("xyz").split()]),
                              "limit": {k: float(v) for k, v in joint.find("limit").attrib.items()}})
            if chain[0]["parent"] != "base_link" or any(chain[i]["parent"] != chain[i-1]["child"] for i in range(1, len(chain))):
                raise ValueError("Expected connected F1 six-joint leg chains rooted at base_link")
            self.legs[side] = chain
            self.lower[side] = np.array([j["limit"]["lower"] for j in chain])
            self.upper[side] = np.array([j["limit"]["upper"] for j in chain])
            vertices = np.unique(collision_vertices(self.urdf, self.robot.find(f"link[@name='{side}_ankle_roll']")), axis=0)
            self.vertices[side] = vertices
            # Collision-mesh bounding-box center, a fixed geometric point.
            self.centers[side] = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        neutral = {s: self.leg_fk(s, np.zeros((1, 6))) for s in SIDES}
        self.neutral_center = {s: neutral[s][0][0] + neutral[s][1][0] @ self.centers[s] for s in SIDES}
        self.neutral_center_span = float(self.neutral_center["left"][1] - self.neutral_center["right"][1])
        self.leg_length = float(np.linalg.norm(neutral["left"][0][0] - self.legs["left"][1]["origin"][:3, 3]
                                              - self.legs["left"][0]["origin"][:3, 3]))

    def leg_indices(self, side, names):
        return [names.index(j["name"]) for j in self.legs[side]]

    def leg_fk(self, side, q, jacobian=False, return_links=False):
        q = np.asarray(q, dtype=float)
        n = len(q)
        p = np.zeros((n, 3))
        r = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        axes, pivots, links = [], [], {}
        for i, joint in enumerate(self.legs[side]):
            o = joint["origin"]
            p = p + apply(r, o[:3, 3])
            r = r @ o[:3, :3]
            axes.append(apply(r, joint["axis"]))
            pivots.append(p.copy())
            r = r @ rot_axis(joint["axis"], q[:, i])
            links[joint["child"]] = (p.copy(), r.copy())
        if return_links:
            return links
        if not jacobian:
            return p, r
        axes, pivots = np.stack(axes, axis=1), np.stack(pivots, axis=1)
        j_position = np.cross(axes, p[:, None, :] - pivots).transpose(0, 2, 1)
        j_orientation = axes.transpose(0, 2, 1)
        return p, r, j_position, j_orientation

    def world_feet(self, qpos, names):
        base_r = quat_rotation(qpos[:, 3:7])
        result = {}
        for side in SIDES:
            p, r = self.leg_fk(side, qpos[:, 7:][:, self.leg_indices(side, names)])
            world_r = base_r @ r
            world_p = qpos[:, :3] + apply(base_r, p)
            result[side] = {"p": world_p, "r": world_r,
                            "center": world_p + apply(world_r, self.centers[side])}
        return result

    def mesh_projections(self, feet, direction):
        result = {}
        for side in SIDES:
            r, p = feet[side]["r"], feet[side]["p"]
            local = apply(np.swapaxes(r, -1, -2), direction)
            # Matrix product evaluates the actual mesh vertices, not a sphere/box approximation.
            values = local @ self.vertices[side].T + np.sum(p * direction, axis=-1, keepdims=True)
            result[side] = (values.min(axis=1), values.max(axis=1))
        return result

    def solve_leg(self, side, initial, target_p, target_r, settings):
        q = np.asarray(initial, dtype=float).copy()
        damping = np.full(len(q), settings["initial_damping"])
        weight = self.leg_length
        iterations = np.zeros(len(q), dtype=int)
        for iteration in range(settings["max_iterations"]):
            p, r, jp, jr = self.leg_fk(side, q, jacobian=True)
            ep, er = target_p-p, orientation_error(r, target_r)
            done = ((np.linalg.norm(ep, axis=-1) <= settings["position_tolerance_m"])
                    & (orientation_angle(r, target_r) <= settings["orientation_tolerance_rad"]))
            if done.all():
                break
            residual = np.concatenate([ep, er*weight], axis=1)
            jac = np.concatenate([jp, jr*weight], axis=1)
            jt = np.swapaxes(jac, -1, -2)
            hessian = jt @ jac + damping[:, None, None]*np.eye(6)
            gradient = apply(jt, residual)
            step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
            step = np.clip(step, -settings["max_joint_step_rad"], settings["max_joint_step_rad"])
            candidate = np.clip(q + step, self.lower[side], self.upper[side])
            cp, cr = self.leg_fk(side, candidate)
            old_cost = np.sum(residual**2, axis=1)
            new_cost = np.sum((target_p-cp)**2, axis=1) + weight**2*np.sum(orientation_error(cr, target_r)**2, axis=1)
            accepted = (new_cost < old_cost) & ~done
            q[accepted] = candidate[accepted]
            damping = np.where(accepted, np.maximum(damping*.5, settings["minimum_damping"]),
                               np.minimum(damping*10, settings["maximum_damping"]))
            iterations[~done] = iteration+1
        p, r = self.leg_fk(side, q)
        pe = np.linalg.norm(target_p-p, axis=-1)
        re = orientation_angle(r, target_r)
        return q, {"position_error_m": pe, "orientation_error_rad": re, "iterations": iterations,
                   "converged": (pe <= settings["position_tolerance_m"]) & (re <= settings["orientation_tolerance_rad"])}
