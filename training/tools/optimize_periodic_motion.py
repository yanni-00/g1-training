"""Jointly fit a periodic ring of root/leg poses with stationary foot plants."""
import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from motion_kinematics import apply, orientation_angle, quat_rotation, rot_z


def optimize_cycle(model, q, names, contacts, fps, nominal, settings):
    n = len(q)
    root_r = quat_rotation(q[:, 3:7])
    root_yaw = np.arctan2(root_r[:, 1, 0], root_r[:, 0, 0])
    legs = {s: np.array(model.leg_indices(s, names))+7 for s in ('left', 'right')}
    initial = np.column_stack([q[:, :3], q[:, legs['left']], q[:, legs['right']]])
    feet = model.world_feet(q, names)
    radius = int(round(settings['cycle_contact_extension_s']*fps))
    contact = {s: maximum_filter1d(contacts[s].astype(int), size=2*radius+1, mode='wrap').astype(bool)
               for s in ('left', 'right')}
    targets = {}
    progress = np.zeros((n, 3))
    progress[:, 0] = nominal*np.arange(n)/fps
    stride = nominal*n/fps
    for side in ('left', 'right'):
        center, rotation = feet[side]['center'].copy(), feet[side]['r'].copy()
        starts = np.flatnonzero(contact[side] & ~np.roll(contact[side], 1))
        if not len(starts):
            raise ValueError('Periodic foot contact mask has no swing/stance transition')
        for start in starts:
            unwrapped = [int(start)]
            while len(unwrapped) < n and contact[side][(unwrapped[-1]+1) % n]:
                unwrapped.append(unwrapped[-1]+1)
            ids = np.array(unwrapped) % n
            cycles = np.array(unwrapped)//n
            world = feet[side]['center'][ids]+progress[ids]
            world[:, 0] += cycles*stride
            anchor = np.median(world, axis=0)
            # In the heading-aligned cycle, planted feet follow the actual
            # travel direction and SSOT neutral orientation, not pelvis twist.
            flat_r = model.leg_fk(side, np.zeros((1, 6)))[1][0]
            relative = (model.vertices[side]-model.centers[side]) @ flat_r.T
            anchor[2] = -relative[:, 2].min()+settings['numeric_sole_clearance_m']
            center[ids] = anchor-progress[ids]
            center[ids, 0] -= cycles*stride
            rotation[ids] = flat_r
        # Connect planted feet with a zero-end-velocity/acceleration swing.
        # Swing clearance comes from the source cycle, not an assumed gait lift.
        swing_starts = np.flatnonzero(~contact[side] & np.roll(contact[side], 1))
        for start in swing_starts:
            unwrapped = [int(start)]
            while len(unwrapped) < n and not contact[side][(unwrapped[-1]+1) % n]:
                unwrapped.append(unwrapped[-1]+1)
            ids = np.array(unwrapped) % n
            cycles = np.array(unwrapped)//n
            prev, following = int(start)-1, unwrapped[-1]+1
            left, right = prev % n, following % n
            anchor_left = center[left]+progress[left]
            anchor_right = center[right]+progress[right]
            anchor_left[0] += (prev//n)*stride
            anchor_right[0] += (following//n)*stride
            t = (np.arange(len(ids))+1)/(len(ids)+1)
            blend = t**3*(10-15*t+6*t*t)
            world = (1-blend[:, None])*anchor_left+blend[:, None]*anchor_right
            apex = max(0., float(feet[side]['center'][ids, 2].max()-min(anchor_left[2], anchor_right[2])))
            world[:, 2] += apex*np.sin(np.pi*t)**2
            swing_r = Slerp([0., .5, 1.], Rotation.from_matrix(np.stack([
                rotation[left], feet[side]['r'][ids[len(ids)//2]], rotation[right]])))(blend).as_matrix()
            local_vertices = model.vertices[side]-model.centers[side]
            minimum = np.einsum('tij,vj->tvi', swing_r, local_vertices)[:, :, 2].min(axis=1)
            world[:, 2] = np.maximum(world[:, 2], -minimum+settings['numeric_sole_clearance_m'])
            center[ids] = world-progress[ids]
            center[ids, 0] -= cycles*stride
            rotation[ids] = swing_r
        weight = np.where(contact[side], 10., 1.)
        targets[side] = {'p': center-apply(rotation, model.centers[side]), 'r': rotation, 'weight': weight}
    lower = np.r_[np.full(3, -np.inf), model.lower['left'], model.lower['right']]
    upper = np.r_[np.full(3, np.inf), model.upper['left'], model.upper['right']]
    # This clamp only constructs a bounded initial iterate. Delivered angles
    # are optimizer solutions and are independently checked, never clipped.
    x0 = np.maximum(np.minimum(initial, upper-1e-10), lower+1e-10)
    pose_weight = settings['cycle_pose_regularization']
    acceleration_weight = settings['cycle_acceleration_regularization']
    velocity_weight = settings['cycle_velocity_constraint_weight']
    limits = np.r_[[j['limit']['velocity'] for j in model.legs['left']],
                   [j['limit']['velocity'] for j in model.legs['right']]]*.99
    t = np.repeat(np.arange(n), 15)
    k = np.tile(np.arange(15), n)
    row = np.arange(n*15)
    acceleration_jac = sparse.csr_matrix((np.r_[np.ones(n*15), -2*np.ones(n*15), np.ones(n*15)]*acceleration_weight,
        (np.tile(row, 3), np.r_[((t-1) % n)*15+k, t*15+k, ((t+1) % n)*15+k])), shape=(n*15, n*15))
    pose_jac = sparse.eye(n*15, format='csr')*pose_weight

    def evaluate(flat, jac=False):
        x = flat.reshape(n, 15)
        residual = np.empty((n, 12))
        blocks = np.zeros((n, 12, 15)) if jac else None
        for side, start, row0 in (('left', 3, 0), ('right', 9, 6)):
            output = model.leg_fk(side, x[:, start:start+6], jacobian=jac)
            p, r = output[:2]
            world_p = x[:, :3]+apply(root_r, p)
            world_r = root_r @ r
            phi = Rotation.from_matrix(targets[side]['r'] @ np.swapaxes(world_r, -1, -2)).as_rotvec()
            w = targets[side]['weight']
            residual[:, row0:row0+3] = (world_p-targets[side]['p'])*w[:, None]
            residual[:, row0+3:row0+6] = phi*model.leg_length*w[:, None]
            if jac:
                jp, jr = output[2:]
                skew = np.zeros((n, 3, 3))
                skew[:, 0, 1], skew[:, 0, 2] = -phi[:, 2], phi[:, 1]
                skew[:, 1, 0], skew[:, 1, 2] = phi[:, 2], -phi[:, 0]
                skew[:, 2, 0], skew[:, 2, 1] = -phi[:, 1], phi[:, 0]
                theta = np.linalg.norm(phi, axis=1)
                coefficient = np.full(n, 1/12)
                mask = theta >= 1e-5
                coefficient[mask] = (1-(theta[mask]/2)/np.tan(theta[mask]/2))/theta[mask]**2
                inverse = np.eye(3)+.5*skew+coefficient[:, None, None]*(skew@skew)
                blocks[:, row0:row0+3, :3] = np.eye(3)*w[:, None, None]
                blocks[:, row0:row0+3, start:start+6] = (root_r@jp)*w[:, None, None]
                blocks[:, row0+3:row0+6, start:start+6] = -(inverse@root_r@jr)*model.leg_length*w[:, None, None]
        velocity = (np.roll(x[:, 3:], -1, axis=0)-x[:, 3:])*fps
        excess = np.maximum(np.abs(velocity)-limits, 0.)
        if jac:
            tt = np.repeat(np.arange(n), 12)
            kk = np.tile(np.arange(12), n)+3
            rr = np.arange(n*12)
            vv = (np.sign(velocity)*(excess > 0)*fps*velocity_weight).reshape(-1)
            velocity_jac = sparse.csr_matrix((np.r_[-vv, vv],
                (np.tile(rr, 2), np.r_[tt*15+kk, ((tt+1) % n)*15+kk])), shape=(n*12, n*15))
            return sparse.vstack([sparse.block_diag(list(blocks), format='csr'), pose_jac, acceleration_jac, velocity_jac], format='csr')
        return np.r_[residual.reshape(-1), ((x-initial)*pose_weight).reshape(-1),
                     ((np.roll(x, 1, axis=0)-2*x+np.roll(x, -1, axis=0))*acceleration_weight).reshape(-1),
                     (excess*velocity_weight).reshape(-1)]
    solution = least_squares(evaluate, x0.reshape(-1), jac=lambda x: evaluate(x, True),
        bounds=(np.tile(lower, n), np.tile(upper, n)), max_nfev=settings['cycle_max_nfev'],
        ftol=1e-10, xtol=1e-10, gtol=1e-10, tr_solver='lsmr')
    solved = solution.x.reshape(n, 15)
    result = q.copy()
    result[:, :3] = solved[:, :3]
    result[:, legs['left']] = solved[:, 3:9]
    result[:, legs['right']] = solved[:, 9:15]
    actual = model.world_feet(result, names)
    report = {'optimizer': 'periodic sparse root XYZ and paired-leg least squares',
              'nfev': int(solution.nfev), 'success': bool(solution.success), 'cost': float(solution.cost),
              'contact_extension_s': settings['cycle_contact_extension_s'], 'feet': {}}
    for side in ('left', 'right'):
        pe = np.linalg.norm(actual[side]['p']-targets[side]['p'], axis=1)
        report['feet'][side] = {'stance_position_target_error_max_m': float(pe[contact[side]].max()),
            'all_frames_position_target_error_max_m': float(pe.max()),
            'stance_orientation_target_error_max_rad': float(orientation_angle(actual[side]['r'], targets[side]['r'])[contact[side]].max())}
    return result, contact, report
