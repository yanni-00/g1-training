"""Audit raw UMR walking and generate traceable F1 pose-level repair candidates.

Never overwrites raw input or marks the result training/native-contact ready.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from motion_kinematics import F1Kinematics, SIDES, apply, quat_rotation, rot_z, wrap

TRAINING = Path(__file__).resolve().parents[1]
PROJECT = TRAINING.parents[2]
sys.path.insert(0, str(TRAINING / "source/f1_beyondmimic"))
from f1_beyondmimic.contract import load_contract, load_json, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(value):
    a = np.asarray(value, dtype=float)
    return {"min": float(a.min()), "p05": float(np.percentile(a, 5)), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def load_motion(path):
    # Object arrays are only read for the colleague-provided joint-name metadata.
    with np.load(path, allow_pickle=True) as z:
        arrays = {key: z[key].copy() for key in z.files}
    q = arrays["qpos"].astype(float)
    names = arrays["robot_joint_names"].reshape(-1).tolist()
    fps = float(arrays["fps"].reshape(-1)[0])
    if q.shape != (len(q), 7+len(names)) or not np.isfinite(q).all() or fps <= 0:
        raise ValueError("Invalid source motion")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate source joint names")
    return arrays, q, names, fps


def heading(q):
    r = quat_rotation(q[:, 3:7])
    yaw = np.arctan2(r[:, 1, 0], r[:, 0, 0])
    lateral = np.stack([-np.sin(yaw), np.cos(yaw), np.zeros(len(q))], axis=1)
    return r, yaw, lateral


def geometry_summary(model):
    zero = {s: model.leg_fk(s, np.zeros((1, 6)), return_links=True) for s in SIDES}
    def span(link):
        return float(zero["left"]["left_"+link][0][0, 1] - zero["right"]["right_"+link][0][0, 1])
    return {"neutral_hip_pitch_pivot_span_m": span("hip_pitch"),
            "neutral_hip_roll_pivot_span_m": span("hip_roll"),
            "neutral_hip_yaw_pivot_span_m": span("hip_yaw"),
            "neutral_ankle_center_span_m": span("ankle_roll"),
            "neutral_collision_mesh_center_span_m": model.neutral_center_span,
            "source": "SSOT URDF joint/origin/axis and scaled foot collision STL; zero-joint FK",
            "human_hip_width": None, "original_mjcf_equivalence_verified": False}


def audit(model, q, names, fps):
    _, yaw, lateral = heading(q)
    feet = model.world_feet(q, names)
    span = np.sum((feet["left"]["center"]-feet["right"]["center"])*lateral, axis=1)
    lateral_extents = model.mesh_projections(feet, lateral)
    clearance = lateral_extents["left"][0] - lateral_extents["right"][1]
    vertical = np.tile([0., 0., 1.], (len(q), 1))
    z_extents = model.mesh_projections(feet, vertical)
    toe, foot_speed = {}, {}
    for side in SIDES:
        foot_yaw = np.arctan2(feet[side]["r"][:, 1, 0], feet[side]["r"][:, 0, 0])
        toe[side] = np.rad2deg(wrap(foot_yaw-yaw))
        foot_speed[side] = np.linalg.norm(np.gradient(feet[side]["center"][:, :2], 1/fps, axis=0, edge_order=2), axis=1)
    position_violations, velocity_violations, max_speed = [], [], {}
    dq = np.gradient(q[:, 7:], 1/fps, axis=0, edge_order=2)
    interval_dq = np.diff(q[:, 7:], axis=0)*fps
    ddq = np.gradient(dq, 1/fps, axis=0, edge_order=2)
    properties = {j.get("name"): j for j in model.joints}
    for i, name in enumerate(names):
        limits = properties[name].find("limit")
        values = q[:, 7+i]
        invalid = (values < float(limits.get("lower"))-1e-6) | (values > float(limits.get("upper"))+1e-6)
        if invalid.any():
            position_violations.append({"joint": name, "frames": np.flatnonzero(invalid).tolist()})
        maximum = max(float(np.abs(dq[:, i]).max()), float(np.abs(interval_dq[:, i]).max()))
        max_speed[name] = maximum
        invalid_v = np.abs(dq[:, i]) > float(limits.get("velocity"))
        invalid_v[1:] |= np.abs(interval_dq[:, i]) > float(limits.get("velocity"))
        if invalid_v.any():
            velocity_violations.append({"joint": name, "frame_count": int(invalid_v.sum()),
                                        "maximum_radps": maximum, "urdf_velocity_radps": float(limits.get("velocity")),
                                        "estimator": "max of central derivative and consecutive-frame difference"})
    summary = {"frame_count": len(q), "fps": fps, "duration_last_minus_first_s": (len(q)-1)/fps,
               "finite": bool(np.isfinite(q).all()),
               "quaternion_norm": stats(np.linalg.norm(q[:, 3:7], axis=1)),
               "foot_mesh_center_lateral_span_m": stats(span),
               "lateral_projected_mesh_edge_clearance_m": stats(clearance),
               "lateral_projection_overlap_frames": int((clearance < 0).sum()),
               "toe_yaw_relative_pelvis_deg": {s: stats(toe[s]) for s in SIDES},
               "sole_min_world_z_m": {s: stats(z_extents[s][0]) for s in SIDES},
               "foot_mesh_center_planar_speed_mps": {s: stats(foot_speed[s]) for s in SIDES},
               "joint_position_limit_violations": position_violations,
               "joint_velocity_limit_violations": velocity_violations,
               "joint_max_abs_velocity_radps": max_speed,
               "global_joint_max_abs_acceleration_radps2": float(np.abs(ddq).max()),
               "finite_clip_joint_endpoint_difference_max_rad": float(np.abs(q[-1, 7:]-q[0, 7:]).max()),
               "contact_state_verified": False,
               "scope": "F1 URDF kinematics only; lateral projection overlap does not prove 3D collision; speeds are finite differences"}
    return summary, {"span": span, "clearance": clearance, "toe": toe, "feet": feet, "z_extents": z_extents}


def repair(model, q, names, settings, fps):
    base_r, yaw, lateral = heading(q)
    source_feet = model.world_feet(q, names)
    reference_yaw = float(np.median(np.unwrap(yaw)))
    fixed_lateral = np.array([-np.sin(reference_yaw), np.cos(reference_yaw), 0.])
    cosine = lateral @ fixed_lateral
    if np.any(cosine <= 0):
        raise ValueError("This clip turns outside a common forward hemisphere; use segmented/contact-aware retargeting")
    span = np.sum((source_feet["left"]["center"]-source_feet["right"]["center"])*lateral, axis=1)
    # The target is an actual robot-geometric reference, not guessed human/robot dimensions.
    target_span = model.neutral_center_span
    low, high = 0., target_span
    while np.median(span+high*cosine) < target_span:
        high *= 2
    for _ in range(60):
        middle = (low+high)/2
        if np.median(span+middle*cosine) < target_span:
            low = middle
        else:
            high = middle
    total_offset = 0. if np.median(span) >= target_span else high
    result = q.copy()
    solver = {}
    target_centers = {}
    target_poses = {}
    base_inv = np.swapaxes(base_r, -1, -2)
    for side, sign in (("left", 1), ("right", -1)):
        foot = source_feet[side]
        old_yaw = np.arctan2(foot["r"][:, 1, 0], foot["r"][:, 0, 0])
        target_r_world = rot_z(wrap(yaw-old_yaw)) @ foot["r"]
        target_center = foot["center"] + sign*.5*total_offset*fixed_lateral
        target_centers[side] = target_center
        target_p_world = target_center - apply(target_r_world, model.centers[side])
        target_r = base_inv @ target_r_world
        target_poses[side] = (target_p_world, target_r)
    # Some standing frames are almost fully extended. A wider/neutral-yaw foot
    # pose then cannot keep the original pelvis height exactly. Solve one common
    # constant lowering, while keeping both world foot-center paths fixed.
    pelvis_lowering = 0.
    pelvis_trials = []
    best_trial = None
    nominal = load_contract()
    for retry in range(settings["pelvis_lowering_max_retries"]+1):
        result[:, 2] = q[:, 2] - pelvis_lowering
        solver = {}
        for side in SIDES:
            target_p_world, target_r = target_poses[side]
            target_p = apply(base_inv, target_p_world-result[:, :3])
            indices = model.leg_indices(side, names)
            # Do not warm-start from a previous straight-knee constrained minimum.
            initial = q[:, 7:][:, indices]
            solved, state = model.solve_leg(side, initial, target_p, target_r, settings["ik"])
            failed = np.flatnonzero(~state["converged"])
            if len(failed):
                nominal_map = dict(zip(nominal["model_joint_names"], nominal["default_joint_pos"]))
                seed = np.array([nominal_map[joint["name"]] for joint in model.legs[side]])
                alternate, alt_state = model.solve_leg(side, np.tile(seed, (len(failed), 1)),
                                                       target_p[failed], target_r[failed], settings["ik"])
                old_cost = state["position_error_m"][failed] + model.leg_length*state["orientation_error_rad"][failed]
                new_cost = alt_state["position_error_m"] + model.leg_length*alt_state["orientation_error_rad"]
                improve = new_cost < old_cost
                selected = failed[improve]
                solved[selected] = alternate[improve]
                for key in state:
                    state[key][selected] = alt_state[key][improve]
            result[:, np.array(indices)+7] = solved
            solver[side] = {"converged_frames": int(state["converged"].sum()),
                            "failed_frames_0based": np.flatnonzero(~state["converged"]).tolist(),
                            "maximum_position_error_m": float(state["position_error_m"].max()),
                            "maximum_orientation_error_rad": float(state["orientation_error_rad"].max()),
                            "maximum_iterations": int(state["iterations"].max())}
        all_converged = all(not solver[s]["failed_frames_0based"] for s in SIDES)
        pelvis_trials.append({"lowering_m": pelvis_lowering, "all_converged": all_converged,
                              "failed_frames": {s: len(solver[s]["failed_frames_0based"]) for s in SIDES},
                              "position_error_max_m": max(solver[s]["maximum_position_error_m"] for s in SIDES)})
        score = sum(len(solver[s]["failed_frames_0based"]) for s in SIDES)
        score = (score, pelvis_trials[-1]["position_error_max_m"])
        if best_trial is None or score < best_trial[0]:
            best_trial = (score, result.copy(), copy.deepcopy(solver), pelvis_lowering)
        elif retry > 0:
            # Refuse unbounded pelvis changes if reachability is not improving.
            _, result, solver, pelvis_lowering = best_trial
            pelvis_trials[-1]["rejected_and_best_restored"] = True
            break
        if all_converged or retry == settings["pelvis_lowering_max_retries"]:
            break
        # The next increment is measured from the previous failed solve, not a
        # guessed morphology/height allowance. Every trial is recorded.
        pelvis_lowering += max(solver[s]["maximum_position_error_m"] for s in SIDES) + settings["ik"]["position_tolerance_m"]
    actual = model.world_feet(result, names)
    extents = model.mesh_projections(actual, np.tile([0., 0., 1.], (len(q), 1)))
    minimum = min(float(extents[s][0].min()) for s in SIDES)
    lift = max(0., -minimum + settings["numeric_only_clearance_tolerance_m"])
    result[:, 2] += lift
    actual = model.world_feet(result, names)
    center_error = {s: float(np.linalg.norm(actual[s]["center"][:, :2]-target_centers[s][:, :2], axis=1).max()) for s in SIDES}
    smoothing = smooth_non_leg_velocity(model, result, names, fps, settings["non_leg_velocity_filter"])
    return result, {"target_center_span_source": "F1 zero-joint URDF FK and scaled collision STL bounding-box centers",
                    "target_median_center_span_m": target_span,
                    "total_constant_world_lateral_offset_m": total_offset,
                    "per_foot_constant_world_lateral_offset_m": total_offset/2,
                    "world_reference_heading_rad": reference_yaw,
                    "constant_pelvis_lowering_m": pelvis_lowering, "pelvis_lowering_trials": pelvis_trials,
                    "constant_root_z_lift_m": lift, "net_root_z_change_m": lift-pelvis_lowering,
                    "source_ground_z_applied": False,
                    "upper_body_joint_values_preserved": not smoothing, "root_xy_and_quaternion_preserved": True,
                    "non_leg_velocity_smoothing": smoothing,
                    "foot_center_horizontal_target_error_max_m": center_error, "ik": solver}


def smooth_non_leg_velocity(model, q, names, fps, settings):
    """Convex local smoothing; only fix actual non-leg URDF speed violations."""
    leg_names = {joint["name"] for side in SIDES for joint in model.legs[side]}
    properties = {joint.get("name"): joint for joint in model.joints}
    weights = np.array(settings["weights"], dtype=float)
    if len(weights) != 3 or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("A convex three-point numerical filter is required")
    changes = []
    for i, name in enumerate(names):
        if name in leg_names:
            continue
        limit = float(properties[name].find("limit").get("velocity"))
        original = q[:, 7+i].copy()
        filtered = original.copy()
        def max_speed(values):
            delivered = values.astype(np.float32).astype(float)
            return max(float(np.abs(np.gradient(delivered, 1/fps, edge_order=2)).max()),
                       float(np.abs(np.diff(delivered)*fps).max()))
        before = max_speed(original)
        if before <= limit:
            continue
        passes = 0
        while max_speed(filtered) > limit and passes < settings["max_passes"]:
            filtered = np.convolve(np.pad(filtered, (1, 1), mode="edge"), weights, mode="valid")
            passes += 1
        if max_speed(filtered) > limit:
            raise ValueError("Cannot satisfy source non-leg velocity bound with local smoothing: " + name)
        q[:, 7+i] = filtered
        changes.append({"joint": name, "before_max_abs_velocity_radps": before,
                        "after_max_abs_velocity_radps": max_speed(filtered), "urdf_velocity_radps": limit,
                        "filter_passes": passes, "max_angle_change_rad": float(np.abs(filtered-original).max())})
    return changes


def metrics_csv(path, before, after, fps):
    import csv
    fields = ["frame", "time_s", "before_span_m", "after_span_m", "before_edge_clearance_m", "after_edge_clearance_m"]
    fields += [f"{kind}_{side}_toe_yaw_deg" for side in SIDES for kind in ("before", "after")]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(len(before["span"])):
            row = {"frame": i, "time_s": i/fps, "before_span_m": before["span"][i], "after_span_m": after["span"][i],
                   "before_edge_clearance_m": before["clearance"][i], "after_edge_clearance_m": after["clearance"][i]}
            for side in SIDES:
                row["before_"+side+"_toe_yaw_deg"] = before["toe"][side][i]
                row["after_"+side+"_toe_yaw_deg"] = after["toe"][side][i]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "f1-retargeting/F1/UMR/weichai_f1_29_accad_walking")
    parser.add_argument("--output", type=Path, required=True, help="New audit directory; refuses replacement")
    parser.add_argument("--candidates", type=Path, required=True, help="New candidate directory; refuses replacement")
    parser.add_argument("--only", help="Optional filename prefix, e.g. B3")
    args = parser.parse_args()
    if args.output.exists() or args.candidates.exists():
        raise ValueError("Use new output/candidate directories; never replace prior evidence")
    args.output.mkdir(parents=True)
    args.candidates.mkdir(parents=True)
    contract = load_contract()
    urdf = TRAINING / contract["urdf"]
    model = F1Kinematics(urdf)
    settings_path = TRAINING / "configs/motion_repair.json"
    settings = load_json(settings_path)
    report = {"format": "f1_walking_motion_repair_audit_v1", "scope": "Kinematic correction candidates only",
              "urdf": str(urdf.resolve()), "urdf_sha256": sha256(urdf),
              "tool_sources_sha256": {name: sha256(Path(__file__).parent/name) for name in
                                      ("repair_walking_motion.py", "motion_kinematics.py", "model_geometry.py")},
              "settings": settings, "settings_sha256": sha256(settings_path), "geometry": geometry_summary(model),
              "source_urdf_equivalence": "Owner designated SSOT as intended F1; original exported MJCF unavailable",
              "human_morphology_causality_verified": False, "native_fk_verified": False,
              "dynamics_verified": False, "owner_visual_acceptance": False, "training_ready": False, "clips": []}
    for path in sorted(args.input.glob("*.npz")):
        if args.only and not path.name.startswith(args.only):
            continue
        source_hash = sha256(path)
        arrays, q, names, fps = load_motion(path)
        if set(names) != set(model.names):
            raise ValueError("Source joint-name set differs from designated F1")
        before, bd = audit(model, q, names, fps)
        corrected, correction = repair(model, q, names, settings, fps)
        # Verify the actual delivered float32 file, not just the float64 solver iterate.
        corrected = corrected.astype(arrays["qpos"].dtype)
        after, ad = audit(model, corrected.astype(float), names, fps)
        kinematic_checks_passed = (all(not correction["ik"][s]["failed_frames_0based"] for s in SIDES)
            and not after["joint_position_limit_violations"] and not after["joint_velocity_limit_violations"]
            and after["lateral_projected_mesh_edge_clearance_m"]["min"] >= 0
            and min(after["sole_min_world_z_m"][s]["min"] for s in SIDES) >= -settings["numeric_only_clearance_tolerance_m"])
        candidate_dir = args.candidates if kinematic_checks_passed else args.output / "rejected"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate = candidate_dir / (path.stem + "_f1_geometry_candidate.npz")
        output_arrays = {key: value.astype(str) if value.dtype.hasobject else value for key, value in arrays.items()}
        output_arrays.update(qpos=corrected, target_urdf=np.array(str(urdf.resolve())),
                             target_urdf_sha256=np.array(sha256(urdf)), source_npz_sha256=np.array(source_hash),
                             repair_format=np.array("f1_urdf_kinematic_candidate_v1"),
                             repair_info_json=np.array(json.dumps(correction)),
                             robot_xml_role=np.array("source provenance only; corrected qpos targets target_urdf"),
                             kinematic_checks_passed=np.array(kinematic_checks_passed),
                             training_ready=np.array(False))
        np.savez_compressed(candidate, **output_arrays)
        if sha256(path) != source_hash:
            raise RuntimeError("Raw input changed")
        row = {"source": str(path.resolve()), "source_sha256": source_hash,
               "candidate": str(candidate.resolve()), "candidate_sha256": sha256(candidate),
               "source_robot_xml": str(arrays["robot_xml"]), "source_ground_z": arrays["ground_z"].tolist(),
               "before": before, "after": after, "correction": correction,
               "all_ik_frames_converged": all(not correction["ik"][s]["failed_frames_0based"] for s in SIDES),
               "kinematic_checks_passed": kinematic_checks_passed,
               "raw_preserved": sha256(path) == source_hash, "training_ready": False}
        report["clips"].append(row)
        metrics_csv(args.output / (path.stem[:2] + "_per_frame.csv"), bd, ad, fps)
        write_json(args.output / "audit.json", report)
        print(json.dumps({"clip": path.name, "all_ik_converged": row["all_ik_frames_converged"],
                          "before_span_median": before["foot_mesh_center_lateral_span_m"]["median"],
                          "after_span_median": after["foot_mesh_center_lateral_span_m"]["median"],
                          "before_toe_median_deg": {s: before["toe_yaw_relative_pelvis_deg"][s]["median"] for s in SIDES},
                          "after_toe_median_deg": {s: after["toe_yaw_relative_pelvis_deg"][s]["median"] for s in SIDES},
                          "position_limit_failures": len(after["joint_position_limit_violations"]),
                          "velocity_limit_failures": len(after["joint_velocity_limit_violations"]),
                          "pelvis_lowering_m": correction["constant_pelvis_lowering_m"],
                          "solver": {s: {k:v for k,v in correction["ik"][s].items() if k != "failed_frames_0based"} for s in SIDES}}), flush=True)


if __name__ == "__main__":
    main()
