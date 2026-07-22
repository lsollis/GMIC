# ============================================================================
# test_ditto_amp_personal.py - the Ditto personal pass follows use_amp
# ----------------------------------------------------------------------------
# The personal pass used to run plain fp32 while the main w pass ran under autocast, so it needed
# ~2x the activation memory of the pass immediately before it -- a hard hardware floor that OOMs
# on T4/L4-class cards. It now follows `use_amp` like every other forward in the executor.
#
# These run on CPU, where autocast is a no-op for memory but still exercises the control flow
# (scaler branch, unscale-before-clip ordering, the non-finite guard, and v still learning).
# The memory claim itself is not testable here; it is pinned by the config + docstring.
#
# Run from gmic_job_hpu/app/custom/:   python -m pytest test_ditto_amp_personal.py -q
# ============================================================================
import contextlib
import copy
import tempfile
from unittest import mock

import torch

from test_executor_integration import (  # noqa: E402  (installs the `resource` stub on import)
    GMIC_PARAMS, make_executor, run_round,
)
from model.gmic import GMIC  # noqa: E402

PROBE = "fusion_dnn.weight"


def _v_probe(ex):
    return dict(ex._v_model.named_parameters())[PROBE].detach().clone()


def test_personal_pass_has_its_own_scaler():
    """v needs a DEDICATED scaler: a shared one would couple the two optimizers' scale schedules.

    Both scalers are None on a CPU-only box (they are built only when cuda is available), which
    would make an identity assertion vacuously pass. Pretend cuda exists for construction so the
    two objects are actually created and can be compared.
    """
    with tempfile.TemporaryDirectory() as td, mock.patch.object(torch.cuda, "is_available",
                                                                return_value=True):
        ex = make_executor("ditto_modulewise", td, use_amp=True)
        assert ex._scaler is not None and ex._v_scaler is not None, "scalers were not constructed"
        assert ex._v_scaler is not ex._scaler, "v must not share the main pass's GradScaler"
    print("[amp] personal pass has a dedicated GradScaler. OK")


def test_personal_pass_trains_under_amp():
    """A use_amp=True Ditto round completes and v actually updates (the B1-style end-to-end guard)."""
    for method in ("ditto", "ditto_modulewise"):
        with tempfile.TemporaryDirectory() as td:
            ex = make_executor(method, td, use_amp=True)
            srv = GMIC(copy.deepcopy(GMIC_PARAMS))
            run_round(ex, srv, round_idx=0, total_rounds=2)
            before = _v_probe(ex)
            run_round(ex, srv, round_idx=1, total_rounds=2)
            after = _v_probe(ex)
            assert torch.isfinite(after).all(), f"{method}: v went non-finite under AMP"
            assert not torch.allclose(before, after), f"{method}: v stopped learning under AMP"
    print("[amp] personal pass trains v under use_amp=True. OK")


def _personal_pass_output_dtypes(ex, srv, use_amp_expected):
    """Run one Ditto round, recording the dtypes v's forward produced in the personal pass.

    Hooks _compute_task_loss, which the personal pass calls with the raw forward outputs. The main
    w pass calls it too, so the recorder is armed only once the personal pass has started.
    """
    seen = []
    original = ex._compute_task_loss

    def recording_loss(outputs, targets):
        if ex._recording:
            stack = [outputs]
            while stack:
                o = stack.pop()
                if torch.is_tensor(o):
                    seen.append(o.dtype)
                elif isinstance(o, dict):
                    stack.extend(o.values())
                elif isinstance(o, (list, tuple)):
                    stack.extend(o)
        return original(outputs, targets)

    ex._recording = False
    ex._compute_task_loss = recording_loss
    original_personal = ex._train_personal_model

    def arming_personal(fl_ctx, abort_signal):
        ex._recording = True
        try:
            return original_personal(fl_ctx, abort_signal)
        finally:
            ex._recording = False

    ex._train_personal_model = arming_personal
    run_round(ex, srv, round_idx=0, total_rounds=2)
    assert seen, "personal pass never reached the loss (nothing recorded)"
    return seen


def test_personal_forward_actually_runs_under_autocast():
    """The behavioral claim: with use_amp=True v's FORWARD runs in reduced precision.

    This is what buys the ~2x activation-memory reduction. Under CPU autocast the reduced dtype is
    bfloat16, so seeing any non-fp32 output dtype proves the forward was inside the autocast
    context -- the old code ran this forward in plain fp32 and would fail here.
    """
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True)
        dtypes = _personal_pass_output_dtypes(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), True)
        assert any(d != torch.float32 for d in dtypes), (
            f"personal forward produced only {set(dtypes)} -- it did NOT run under autocast, so it "
            f"still costs full fp32 activation memory")
    print(f"[amp] personal forward runs under autocast (dtypes={set(dtypes)}). OK")


def test_personal_forward_is_fp32_when_amp_off():
    """Mirror image: use_amp=False must still produce a genuinely fp32 forward."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=False)
        dtypes = _personal_pass_output_dtypes(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), False)
        assert all(d == torch.float32 for d in dtypes), \
            f"use_amp=False personal forward was not fp32 (got {set(dtypes)})"
    print("[amp] personal forward is fp32 when use_amp=False. OK")


def test_stage_sync_does_not_change_training():
    """stage_sync only inserts torch.cuda.synchronize(); v must still train identically.

    Guards the diagnostic against becoming a behavior change: syncing is a scheduling barrier,
    never a numerical one, so the same seed must give the same v.
    """
    probes = {}
    for flag in (False, True):
        with tempfile.TemporaryDirectory() as td:
            torch.manual_seed(1234)
            ex = make_executor("ditto_modulewise", td, use_amp=True, stage_sync=flag)
            torch.manual_seed(1234)
            run_round(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), round_idx=0, total_rounds=1)
            probes[flag] = _v_probe(ex)
    assert torch.allclose(probes[False], probes[True], atol=1e-6), \
        "stage_sync altered v -- a diagnostic must not change the math"
    print("[amp] stage_sync is behavior-neutral. OK")


def test_personal_pass_stages_are_granular_under_amp():
    """The AMP-only calls must each be individually nameable when something wedges.

    A hang inside scaler.unscale_ vs clip vs scaler.step is the exact question we cannot answer
    today, so these must be distinct stages rather than one lumped 'step'.
    """
    import bc_executor as BE
    seen, holder = [], {}
    real_hb = BE.GMICFederatedExecutor._heartbeat

    @contextlib.contextmanager
    def capturing_hb(self, label, state):
        holder["state"] = state  # the dict the loop mutates; read after each _sync
        with real_hb(self, label, state) as s:
            yield s

    with tempfile.TemporaryDirectory() as td:
        # The AMP branch is gated on `scaler is not None`, and the scaler is only built when CUDA
        # is available -- so on a CPU-only box the fp32 branch would run and the AMP-only stages
        # would never appear. Fake CUDA for construction: PyTorch then builds a GradScaler that
        # self-disables (no CUDA), which is exactly the pass-through we want for a label test.
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            ex = make_executor("ditto_modulewise", td, use_amp=True, heartbeat_interval_s=0)
        assert ex._v_scaler is not None
        real_sync = ex._sync
        ex._sync = lambda: (seen.append(holder.get("state", {}).get("stage")), real_sync())[1]
        BE.GMICFederatedExecutor._heartbeat = capturing_hb
        try:
            run_round(ex, GMIC(copy.deepcopy(GMIC_PARAMS)), round_idx=0, total_rounds=1)
        finally:
            BE.GMICFederatedExecutor._heartbeat = real_hb
    for expected in ("forward", "backward", "unscale", "clip", "step", "scaler-update"):
        assert expected in seen, f"stage '{expected}' is not separately observable; got {set(seen)}"
    print(f"[amp] personal step exposes granular stages: {sorted(set(x for x in seen if x))}. OK")


def test_fp32_personal_pass_still_supported():
    """use_amp=False keeps the original fp32 path (sim jobs / A100 reruns are unaffected)."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=False)
        srv = GMIC(copy.deepcopy(GMIC_PARAMS))
        run_round(ex, srv, round_idx=0, total_rounds=2)
        before = _v_probe(ex)
        run_round(ex, srv, round_idx=1, total_rounds=2)
        after = _v_probe(ex)
        assert torch.isfinite(after).all()
        assert not torch.allclose(before, after), "v stopped learning in the fp32 path"
    print("[amp] fp32 personal pass unchanged when use_amp=False. OK")


def test_nonfinite_loss_still_skips_step():
    """The persist-across-rounds poison guard must survive the AMP refactor."""
    with tempfile.TemporaryDirectory() as td:
        ex = make_executor("ditto_modulewise", td, use_amp=True)
        srv = GMIC(copy.deepcopy(GMIC_PARAMS))
        run_round(ex, srv, round_idx=0, total_rounds=2)  # creates v + ref
        before = _v_probe(ex)
        # Force every task loss non-finite; v must be left exactly as-is, not poisoned.
        ex._compute_task_loss = lambda outputs, targets: torch.tensor(float("nan"))
        ex._train_personal_model(__import__("nvflare.apis.fl_context", fromlist=["FLContext"]).FLContext(),
                                 __import__("nvflare.apis.signal", fromlist=["Signal"]).Signal())
        after = _v_probe(ex)
        assert torch.equal(before, after), "non-finite loss poisoned v (guard broken)"
    print("[amp] non-finite loss skips the step; v untouched. OK")


if __name__ == "__main__":
    test_personal_pass_has_its_own_scaler()
    test_personal_forward_actually_runs_under_autocast()
    test_personal_forward_is_fp32_when_amp_off()
    test_personal_pass_trains_under_amp()
    test_fp32_personal_pass_still_supported()
    test_nonfinite_loss_still_skips_step()
    print("all AMP personal-pass tests passed")
