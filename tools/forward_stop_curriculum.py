"""The owner's forward/GO/STOP scope; no new actor inputs or reference reset."""
FORMAT='f1_forward_stop_curriculum_v1'
SPEC={'kind':FORMAT,'speed_scale_range':[1.0,1.0],'yaw_rate_range':[0.0,0.0],
      'straight_probability':1.0,'initial_go_probability':0.0}

def configure(cfg,spec):
    if spec!=SPEC:raise ValueError('Unregistered forward/stop curriculum')
    command=cfg.commands.motion
    command.speed_scale_range=tuple(spec['speed_scale_range'])
    command.yaw_rate_range=tuple(spec['yaw_rate_range'])
    command.straight_probability=spec['straight_probability']
    command.initial_go_probability=spec['initial_go_probability']
    # Original GO/STOP dwell times, random support selection, reference phase
    # braking, episode length, controls, observations and guards stay intact.
    return spec

def audit_live(command,spec):
    if spec!=SPEC:raise ValueError('Wrong actual curriculum specification')
    cfg=command.cfg
    for name in ('speed_scale_range','yaw_rate_range'):
        if list(getattr(cfg,name))!=spec[name]:raise ValueError('Native curriculum mismatch: '+name)
    for name in ('straight_probability','initial_go_probability'):
        if getattr(cfg,name)!=spec[name]:raise ValueError('Native curriculum mismatch: '+name)
    if tuple(cfg.go_duration_range)!=(4.0,12.0) or tuple(cfg.stop_duration_range)!=(2.0,4.0):
        raise ValueError('Unregistered dwell-time change')
    nominal=command.reference.nominal_speed
    expected=((0.0,nominal),(0.0,0.0),(0.0,0.0))
    if tuple(tuple(x) for x in command.command_limits)!=expected:raise ValueError('Native command limits mismatch')
    return {**spec,'native_environments':command.num_envs,'nominal_reference_speed_mps':nominal,
            'command_limits':[list(x) for x in expected],
            'go_duration_range_s':list(cfg.go_duration_range),'stop_duration_range_s':list(cfg.stop_duration_range),
            'scope':'Stationary episode starts; original GO/STOP sampling at fixed forward speed; no yaw/backward/lateral curriculum',
            'unchanged':'actor, original reference/phase clock, PD, assets, rewards and terminations'}
