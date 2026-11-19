"""A training-only cost for prematurely losing the reference support foot."""
import math

from forward_stop_curriculum import SPEC as FORWARD_STOP, audit_live, configure

TERM_NAME = 'preswing_support_loss'
FEET = ('left_ankle_roll', 'right_ankle_roll')
SPEC = {
    'kind': 'f1_preswing_support_loss_v1',
    'forward_stop': FORWARD_STOP,
    'cycle_frames': 63,
    # The rising edge comes from the paired ankle diagnosis. The falling edge
    # is the actual first swing frame in each side's saved source contact mask.
    'windows': [[18.5, 35.0], [50.5, 67.0]],
    'weight': -0.2,
    'contact_threshold_N': 10.0,
}


def support_loss_cost(phase, speed, rate, normal_force, *, windows, period, threshold):
    """Soft force deficit in a tapered support-only window for each foot."""
    import torch

    start = phase.new_tensor([window[0] for window in windows])
    width = phase.new_tensor([window[1]-window[0] for window in windows])
    elapsed = torch.remainder(phase[:, None]-start, period)
    fraction = (elapsed/width).clamp(0.0, 1.0)
    taper = torch.where(elapsed < width, torch.sin(math.pi*fraction).square(), 0.0)
    deficit = (1.0-normal_force/threshold).clamp(0.0, 1.0)
    nominal_go = (speed>0.0) & (rate==1.0)
    return (taper*deficit).sum(dim=-1)*nominal_go


def preswing_support_loss(env, sensor_cfg, windows, period, threshold):
    command = env.command_manager.get_term('motion')
    reference = command.reference
    force_z = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return support_loss_cost(reference.clock.phase, reference.bounded_command[:,0],
        reference.clock.rate, force_z, windows=windows, period=period, threshold=threshold)


def configure_training(cfg, spec):
    if spec != SPEC:
        raise ValueError('Unregistered training reward delta')
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

    configure(cfg, spec['forward_stop'])
    if hasattr(cfg.rewards, TERM_NAME):
        raise ValueError('Duplicate reward term')
    setattr(cfg.rewards, TERM_NAME, RewardTermCfg(
        func=preswing_support_loss, weight=spec['weight'],
        params={'sensor_cfg':SceneEntityCfg('contact_forces',body_names=list(FEET),preserve_order=True),
                'windows':spec['windows'],'period':spec['cycle_frames'],'threshold':spec['contact_threshold_N']}))


def audit_training(env, spec):
    import torch

    if spec != SPEC:
        raise ValueError('Unexpected reward delta')
    base = env.unwrapped
    command = base.command_manager.get_term('motion')
    curriculum = audit_live(command, spec['forward_stop'])
    if command.reference.clock.period != spec['cycle_frames']:
        raise ValueError('Native gait period mismatch')
    if TERM_NAME not in base.reward_manager.active_terms:
        raise ValueError('Training reward missing from active native manager')
    term = base.reward_manager.get_term_cfg(TERM_NAME)
    params = term.params
    sensor_cfg = params['sensor_cfg']
    sensor = base.scene.sensors[sensor_cfg.name]
    expected = [sensor.body_names.index(name) for name in FEET]
    ids = sensor_cfg.body_ids
    if (term.func is not preswing_support_loss or term.weight != spec['weight']
        or list(ids) != expected or params['windows'] != spec['windows']
        or params['period'] != spec['cycle_frames'] or params['threshold'] != spec['contact_threshold_N']):
        raise ValueError('Live phase-local reward parameters differ from registered source')
    value = term.func(base, **params)
    if value.shape != (base.num_envs,) or not torch.isfinite(value).all() or (value<0).any():
        raise ValueError('Invalid native reward output')
    return curriculum, {'name':TERM_NAME,'weight':term.weight,'sensor_bodies':list(FEET),
        'sensor_body_indices':expected,'value_shape':list(value.shape),'finite_nonnegative':True,
        'windows':params['windows'],'period':params['period'],'threshold':params['threshold']}
