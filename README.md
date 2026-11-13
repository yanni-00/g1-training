# F1 full-body walking — G1 original PPO route
Branch: codex/f1-29dof-walking, repository yanni-00/g1-training.
Current owner request: configs/owner_training_profile.json (PD; friction=0, damping=0.05; no added identification/armature/filter).
Frozen effect reference: configs/reference/g1_11k_manifest.json, model_10999_factorygait.pt. It is not an F1 initialization checkpoint.
The lossless F1 asset archive is restored and hash-verified by the native entry before loading Isaac. Source SSOT meshes are unchanged.
Current registered bounded probe: configs/run_request.json. It validates actual training/new-checkpoint delivery/load/feedback; it is not the final walking result.
GM entry: gm-run g1-training/scripts/rsl_rl/gm_entry.py
Installed Isaac Lab/Sim and RSL-RL are required. No credentials are stored here. Scope/acceptance: .oma/requirements.md.
