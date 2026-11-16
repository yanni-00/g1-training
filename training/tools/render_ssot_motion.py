"""Render actual SSOT geometry in front, side and foot top views; no dynamics."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_ssot_retarget import native_model, sha, write_json
from repair_walking_motion import load_motion


def add_world_grid(scene, ground_z, bounds, spacing):
    """World-fixed visual decoration only; never changes model or contact geometry."""
    xmin, xmax, ymin, ymax = bounds
    for axis, lo, hi in ((0, xmin, xmax), (1, ymin, ymax)):
        for index in range(int(np.floor(lo/spacing)), int(np.ceil(hi/spacing))+1):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('Diagnostic grid exceeds renderer scene capacity')
            position = np.array([(xmin+xmax)/2, (ymin+ymax)/2, ground_z+.0002])
            position[axis] = index*spacing
            size = np.array([(xmax-xmin)/2, (ymax-ymin)/2, .00015])
            size[axis] = .0007
            color = [.25, .25, .25, 1.] if index % 5 == 0 else [.55, .55, .55, 1.]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                size, position, np.eye(3).reshape(-1), np.array(color, dtype=np.float32))
            scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--repeat-cycles', type=int, default=1)
    parser.add_argument('--diagnostic', action='store_true', help='Fixed world side camera, feet close-up, and world floor grid')
    parser.add_argument('--grid-spacing', type=float, default=.1, help='Visual grid spacing in metres; no change to motion')
    parser.add_argument('--render-fps', type=float, default=15., help='Maximum sampled source frames per second')
    parser.add_argument('--playback-rate', type=float, default=1., help='Video speed relative to source time')
    args = parser.parse_args()
    if args.grid_spacing <= 0 or args.render_fps <= 0 or args.playback_rate <= 0:
        raise ValueError('Positive grid spacing, render FPS and playback rate required')
    if args.output.exists():
        raise FileExistsError('Refuse to replace a preview')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays, q, names, fps = load_motion(args.input)
    unique_frames = len(q)
    if 'cycle_frames' in arrays:
        if args.repeat_cycles < 1:
            raise ValueError('Positive repeat count required')
        q = np.tile(q, (args.repeat_cycles, 1))
        q[:, 0] += float(arrays['nominal_reference_speed_mps'][0])*np.arange(len(q))/fps
    model = native_model(args.model.resolve())
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = [.6, .6, .6]
    model.vis.headlight.diffuse[:] = [.8, .8, .8]
    addresses = [model.joint(name).qposadr[0] for name in names]
    for i in range(model.ngeom):
        if model.geom_group[i] == 1:
            body = model.body(int(model.geom_bodyid[i])).name
            if not any(part in body for part in ('hip_', 'knee_', 'ankle_')):
                model.geom_group[i] = 3
            elif body in {'left_ankle_roll', 'right_ankle_roll'}:
                model.geom_group[i] = 4
            if body.startswith('left_'):
                model.geom_rgba[i] = [.2, .5, .85, 1.]
            elif body.startswith('right_'):
                model.geom_rgba[i] = [.9, .45, .15, 1.]
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    option.geomgroup[1:4] = 1
    option.geomgroup[4] = 1
    renderer = mujoco.Renderer(model, height=400, width=480)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 19) if font_path.is_file() else ImageFont.load_default()
    step = max(1, int(np.ceil(fps/args.render_fps))) if args.diagnostic else max(1, int(round(fps/args.render_fps)))
    indices = list(range(0, len(q), step))
    if args.max_frames:
        indices = indices[:args.max_frames]
    writer = imageio.get_writer(args.output, fps=fps/step*args.playback_rate, codec='libx264', quality=8, macro_block_size=1)
    fixed_center = np.median(q[:, :3], axis=0)+model.stat.center
    fixed_distance = max(model.stat.extent*1.2, model.stat.extent+np.ptp(q[:, 0])*.8)
    margin = model.stat.extent
    grid_bounds = (q[:, 0].min()-margin, q[:, 0].max()+margin,
                   q[:, 1].min()-margin, q[:, 1].max()+margin)
    ground_z = float(model.geom('ground').pos[2])
    clearance = None
    if args.diagnostic:
        from motion_kinematics import F1Kinematics
        contract = json.loads((Path(__file__).resolve().parents[1]/'configs/robot_contract.json').read_text(encoding='utf-8'))
        kin = F1Kinematics(Path(__file__).resolve().parents[1]/contract['urdf'])
        feet = kin.world_feet(q, names)
        clearance = kin.mesh_projections(feet, np.tile([0., 0., 1.], (len(q), 1)))
    titles = ('FIXED WORLD SIDE', 'FOLLOWING FEET SIDE', 'FOLLOWING FEET TOP') if args.diagnostic else ('FRONT', 'SIDE', 'FOOT TOP')
    sample = None
    for output_frame, frame in enumerate(indices):
        row = q[frame]
        data.qpos[:7] = row[:7]
        data.qpos[addresses] = row[7:]
        mujoco.mj_forward(model, data)
        quat = row[3:7]
        yaw = np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]), 1-2*(quat[2]**2+quat[3]**2))
        canvas = Image.new('RGB', (1440, 480 if args.diagnostic else 440), 'white')
        draw = ImageDraw.Draw(canvas)
        for panel, (azimuth, elevation) in enumerate(((90, -8), (90, -8), (0, -89)) if args.diagnostic else ((0, -8), (90, -8), (0, -89))):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = row[:3]+model.stat.center
            camera.distance = model.stat.extent*1.2
            camera.azimuth = np.rad2deg(yaw)+azimuth
            camera.elevation = elevation
            option.geomgroup[3] = int(panel != 2)
            option.geomgroup[1] = int(panel != 2)
            if panel == 2:
                left, right = model.body('left_ankle_roll').id, model.body('right_ankle_roll').id
                camera.lookat[:] = (data.xpos[left]+data.xpos[right])/2
                camera.distance = model.stat.extent*.75
            if args.diagnostic and panel == 0:
                camera.lookat[:] = fixed_center
                camera.distance = fixed_distance
                camera.azimuth = 90
            if args.diagnostic and panel == 1:
                left, right = model.body('left_ankle_roll').id, model.body('right_ankle_roll').id
                camera.lookat[:] = (data.xpos[left]+data.xpos[right])/2
                camera.lookat[2] = ground_z+kin.leg_length*.3
                camera.distance = model.stat.extent*.62
                option.geomgroup[3] = 0
            renderer.update_scene(data, camera=camera, scene_option=option)
            if args.diagnostic:
                add_world_grid(renderer.scene, ground_z, grid_bounds, args.grid_spacing)
            pixels = renderer.render()
            canvas.paste(Image.fromarray(pixels), (panel*480, 40))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 1440, 39), fill='white')
        for panel, title in enumerate(titles):
            draw.text((panel*480+12, 10), title, fill='black', font=font)
        if args.diagnostic:
            draw.text((12, 450), f'Source {frame/fps:.2f} s  |  {args.playback_rate:g}x speed  |  grid {args.grid_spacing:g} m  |  unchanged NPZ', fill='black', font=font)
            for j, side in enumerate(('left', 'right')):
                state = 'STANCE' if arrays.get('contact_'+side, np.zeros(unique_frames, dtype=bool))[frame % unique_frames] else 'SWING'
                label = f'{side.upper()}: {state}  sole {clearance[side][0][frame]*1000:.1f} mm'
                draw.text((655+j*390, 450), label, fill='black', font=font)
        else:
            draw.text((120, 10), 'SSOT kinematic replay', fill='black', font=font)
            draw.text((1160, 10), f'{frame/fps:.2f} s', fill='black', font=font)
        writer.append_data(np.asarray(canvas))
        if output_frame == len(indices)//2:
            sample = canvas.copy()
        if output_frame % 30 == 0:
            print('[SSOT render]', output_frame, '/', len(indices), flush=True)
    writer.close()
    renderer.close()
    if sample:
        sample.save(args.output.with_suffix('.png'))
    write_json(args.output.with_suffix('.json'), {'input': str(args.input.resolve()), 'input_sha256': sha(args.input),
        'model_sha256': sha(args.model), 'video_sha256': sha(args.output), 'rendered_frames': len(indices),
        'video_fps': fps/step*args.playback_rate, 'native_geometry': True, 'dynamic_simulation': False,
        'renderer_sha256': sha(Path(__file__)), 'diagnostic_views': args.diagnostic,
        'playback_rate': args.playback_rate, 'source_fps': fps, 'source_frame_indices': indices,
        'world_fixed_grid_spacing_m': args.grid_spacing if args.diagnostic else None,
        'cycle_world_progress_reconstructed': 'cycle_frames' in arrays,
        'motion_arrays_modified': False})


if __name__ == '__main__':
    main()
