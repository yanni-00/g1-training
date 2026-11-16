"""Diagnostic URDF-FK lower-body/foot-mesh projections, not dynamic simulation."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motion_kinematics import F1Kinematics, SIDES, apply, rot_z, wrap
from repair_walking_motion import load_motion, heading, TRAINING, write_json

sys.path.insert(0, str(TRAINING / "source/f1_beyondmimic"))
from f1_beyondmimic.contract import load_contract, load_json, sha256


def font(size):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def draw_frame(model, original, candidate, names, index, fps, width, height, top_bounds):
    canvas = np.full((height, width, 3), (245, 247, 250), dtype=np.uint8)
    panel_width = width // 2
    colors = {"left": (33, 101, 216), "right": (225, 127, 39)}
    rows = []
    for panel, (label, data) in enumerate((("ORIGINAL", original), ("F1 KINEMATIC CANDIDATE", candidate))):
        row = data[index:index+1]
        base_r, yaw, _ = heading(row)
        frame_rotation = rot_z(-yaw)[0]
        center = row[0, :3].copy()
        center[2] = 0
        feet = model.world_feet(row, names)
        x0 = panel*panel_width
        def body_coordinates(points):
            return (points-center) @ frame_rotation.T
        def front(points):
            return np.stack([x0+panel_width/2-points[:, 1]*680, 380-points[:, 2]*300], axis=1).round().astype(np.int32)
        def top(points):
            lower, upper = top_bounds
            py = 440 + (upper-points[:, 0])/(upper-lower)*(height-490)
            return np.stack([x0+panel_width/2-points[:, 1]*680, py], axis=1).round().astype(np.int32)
        cv2.rectangle(canvas, (x0+10, 50), (x0+panel_width-10, height-15), (255, 255, 255), -1)
        cv2.line(canvas, (x0+30, 380), (x0+panel_width-30, 380), (180, 188, 198), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0+panel_width//2, 420), (x0+panel_width//2, height-35), (210, 217, 227), 1)
        hip_points = []
        toe = {}
        for side in SIDES:
            indices = model.leg_indices(side, names)
            links = model.leg_fk(side, row[:, 7:][:, indices], return_links=True)
            keys = [side+"_hip_roll", side+"_hip_yaw", side+"_knee_pitch", side+"_ankle_roll"]
            points = np.stack([links[key][0][0] for key in keys])
            world_points = points @ base_r[0].T + row[0, :3]
            projected = front(body_coordinates(world_points))
            hip_points.append(projected[0])
            cv2.polylines(canvas, [projected], False, colors[side], 7, cv2.LINE_AA)
            for point in projected:
                cv2.circle(canvas, tuple(point), 6, colors[side], -1, cv2.LINE_AA)
            foot = feet[side]
            vertices = model.vertices[side] @ foot["r"][0].T + foot["p"][0]
            for projector in (front, top):
                hull = cv2.convexHull(projector(body_coordinates(vertices)))
                cv2.fillConvexPoly(canvas, hull, tuple(int(c*.4+255*.6) for c in colors[side]), cv2.LINE_AA)
                cv2.polylines(canvas, [hull], True, colors[side], 2, cv2.LINE_AA)
            local_tip = model.centers[side].copy()
            local_tip[0] = model.vertices[side][:, 0].max()
            arrow_world = np.stack([foot["center"][0], foot["p"][0]+foot["r"][0] @ local_tip])
            arrow = top(body_coordinates(arrow_world))
            cv2.arrowedLine(canvas, tuple(arrow[0]), tuple(arrow[1]), colors[side], 3, cv2.LINE_AA, tipLength=.3)
            toe[side] = float(np.rad2deg(wrap(np.arctan2(foot["r"][0, 1, 0], foot["r"][0, 0, 0])-yaw[0])))
        cv2.line(canvas, tuple(hip_points[0]), tuple(hip_points[1]), (60, 75, 85), 8, cv2.LINE_AA)
        relative = body_coordinates(feet["left"]["center"]-feet["right"]["center"]+center)
        rows.append((label, float(relative[0, 1]), toe))
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((20, 12), f"F1 URDF-FK comparison | frame {index} | t={index/fps:.3f}s", fill=(25, 35, 45), font=font(22))
    for panel, (label, span, toe) in enumerate(rows):
        x0 = panel*panel_width
        draw.text((x0+25, 57), label, fill=(170, 50, 45) if panel == 0 else (25, 125, 90), font=font(22))
        draw.text((x0+25, 88), f"Sole-center span: {span*100:.2f} cm", fill=(30, 40, 50), font=font(20))
        draw.text((x0+25, 115), f"Toe yaw L/R: {toe['left']:+.2f} / {toe['right']:+.2f} deg", fill=(30, 40, 50), font=font(20))
        draw.text((x0+25, 150), "FRONT PROJECTION", fill=(100, 110, 120), font=font(15))
        draw.text((x0+25, 401), "TOP VIEW: forward is up; blue=left / orange=right", fill=(70, 80, 90), font=font(15))
    draw.text((20, height-24), "Collision-mesh silhouettes + skeleton. Pose-level playback only; no physics/contact/dynamic acceptance.", fill=(90, 100, 110), font=font(15))
    return pil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new preview directory")
    args.output.mkdir(parents=True)
    report = load_json(args.audit)
    c = load_contract()
    model = F1Kinematics(TRAINING / c["urdf"])
    settings = report["settings"]
    outputs = []
    for entry in report["clips"]:
        if not entry.get("kinematic_checks_passed"):
            continue
        _, original, names, fps = load_motion(entry["source"])
        _, candidate, candidate_names, _ = load_motion(entry["candidate"])
        if names != candidate_names or len(original) != len(candidate):
            raise ValueError("Preview before/after identity mismatch")
        bounds = []
        for data in (original, candidate):
            _, yaw, _ = heading(data)
            forward = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(len(data))], axis=1)
            extents = model.mesh_projections(model.world_feet(data, names), forward)
            root_projection = np.sum(data[:, :3]*forward, axis=1)
            bounds.extend((float((extents[s][0]-root_projection).min()), float((extents[s][1]-root_projection).max())) for s in SIDES)
        lower = min(v[0] for v in bounds)
        upper = max(v[1] for v in bounds)
        padding = (upper-lower)*.05
        top_bounds = (lower-padding, upper+padding)
        prefix = Path(entry["source"]).stem[:2]
        video = args.output / (prefix+"_original_vs_candidate.mp4")
        width, height, render_fps = settings["render_width"], settings["render_height"], settings["render_fps"]
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MP4 writer unavailable")
        frames = np.rint(np.arange(0, (len(original)-1)/fps+1e-9, 1/render_fps)*fps).astype(int)
        try:
            for index in frames:
                im = draw_frame(model, original, candidate, names, int(index), fps, width, height, top_bounds)
                writer.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        still = args.output / (prefix+"_comparison_middle_frame.png")
        draw_frame(model, original, candidate, names, len(original)//2, fps, width, height, top_bounds).save(still)
        cap = cv2.VideoCapture(str(video))
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        cap.release()
        if decoded != len(frames):
            raise RuntimeError("Preview encode/decode frame mismatch")
        outputs.append({"clip": prefix, "video": str(video.resolve()), "video_sha256": sha256(video),
                        "render_fps": render_fps, "decoded_frame_count": decoded,
                        "still": str(still.resolve()), "scope": "URDF FK projection; not native physics"})
        print(json.dumps({"clip": prefix, "decoded_frames": decoded, "video": str(video)}), flush=True)
    write_json(args.output / "preview_manifest.json", {"audit_sha256": sha256(args.audit),
                                                       "renderer_sha256": sha256(__file__), "outputs": outputs})


if __name__ == "__main__":
    main()
