from rsl_rl.runners import OnPolicyRunner
from .checkpoint import collect_runtime
from .runtime_policy import verify_runtime_checkpoint


class F1OnPolicyRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        info = dict(infos or {})
        # Do not reuse a G1 cached runtime; the live native environment is authoritative.
        if not hasattr(self, "_f1_runtime"):
            self._f1_runtime = collect_runtime(self.env, self.alg.policy)
        info["f1_runtime"] = self._f1_runtime
        super().save(path, info)
        if not getattr(self, "_first_save_verified", False):
            print("[F1] normalized actor parity:", verify_runtime_checkpoint(self.alg.policy, path), flush=True)
            self._first_save_verified = True
