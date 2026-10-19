import os, subprocess, pathlib
print("=== CWD ===", os.getcwd())
print("=== find model_*.pt in likely roots ===")
for root in ["/workspace/isaaclab", "/workspace", "/personal", "/root", "/mnt/data", "/data"]:
    if not os.path.isdir(root): continue
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "logs", "source", "isaaclab", "_isaac_sim")]
        for f in files:
            if f.startswith("model_") and f.endswith(".pt"):
                print(os.path.join(dirpath, f))
        if dirpath.count(os.sep) > 6: dirs[:] = []
print("=== ls g1-training root ===")
gt = "/workspace/isaaclab/g1-training"
if os.path.isdir(gt):
    print(sorted(os.listdir(gt)))
