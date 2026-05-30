# ============================================================================
# test_fl_methods.py - CPU synthetic tests for the federated-method primitives
# ----------------------------------------------------------------------------
# Run from gmic_job/app/custom/:   python test_fl_methods.py
# (or with pytest:                 pytest -q test_fl_methods.py)
#
# Covers brief tests:
#   1. Module-grouping assertion (every param -> exactly one group)
#   2. BN-skip (FedBN load preserves local BN; non-BN equals global)
#   3. Proximal-term (zero at ref; scales with per-group lambda)
#   7. Pretrained-load (released GMIC ckpt loads essentially complete)
#   + model forward smoke on a synthetic full-size input
#   + Ditto personal-model persistence (v not reset to global across rounds)
#
# Tests 4 (full-executor per-method round) and 6 (no-regression vs old executor)
# require NVFLARE + a data loader and are exercised by the companion
# integration check; see report. The math those rely on is covered here.
# ============================================================================
import os
import sys
import copy

import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from model.gmic import GMIC  # noqa: E402
import fl_utils as F  # noqa: E402

GMIC_PARAMS = {
    "device_type": "cpu", "gpu_number": 0,
    "max_crop_noise": (100, 100), "max_crop_size_noise": 100,
    "image_path": "x", "cam_size": (46, 30), "K": 6,
    "crop_shape": (256, 256), "post_processing_dim": 256,
    "num_classes": 2, "use_v1_global": False,
    "percent_t": 0.02, "lambda_l1": 1e-5,
}

CKPT = os.path.normpath(os.path.join(
    THIS_DIR, "..", "..", "..", "site_folders", "Moffitt", "models", "sample_model_5.p"
))


def build_model():
    return GMIC(copy.deepcopy(GMIC_PARAMS))


# --------------------------------------------------------------------------
def test_grouping():
    m = build_model()
    sizes = F.assert_grouping_complete(m)
    total_params = sum(1 for _ in m.named_parameters())
    print(f"[grouping] sizes={sizes} total={total_params}")
    assert sum(sizes.values()) == total_params, "some params unassigned"
    assert sizes["global"] == 69 and sizes["local"] == 60 and sizes["fusion"] == 6, sizes
    # every key resolves without raising; buffers map to a known/ignore group
    for name, _ in m.named_buffers():
        F.module_group(name)
    print("[grouping] OK")


def test_bn_skip():
    g = build_model()
    m = build_model()
    bn_key = "ds_net.final_bn.running_mean"          # a BN buffer (skipped)
    conv_key = "ds_net.first_conv.weight"            # a non-BN param (loaded)
    assert bn_key in g.state_dict() and conv_key in g.state_dict()

    with torch.no_grad():
        g.state_dict()[bn_key].zero_()
        g.state_dict()[conv_key].zero_()
        m.state_dict()[bn_key].fill_(7.0)
        m.state_dict()[conv_key].fill_(7.0)

    skip = F.bn_state_keys(m)
    assert bn_key in skip and conv_key not in skip
    print(f"[bn_skip] #BN keys skipped = {len(skip)}")

    stats = F.load_global_into(m, g.state_dict(), skip_keys=skip)
    print(f"[bn_skip] load stats = {stats}")

    # local BN preserved (still 7), differs from global (0)
    assert torch.allclose(m.state_dict()[bn_key], torch.full_like(m.state_dict()[bn_key], 7.0))
    assert not torch.allclose(m.state_dict()[bn_key], g.state_dict()[bn_key])
    # non-BN conv now equals global (0)
    assert torch.allclose(m.state_dict()[conv_key], g.state_dict()[conv_key])
    print("[bn_skip] OK")


def test_proximal():
    m = build_model()
    ref = F.clone_detached_state(m)

    # zero at the reference
    p0 = F.proximal_penalty(m, ref, 1.0)
    assert float(p0) < 1e-9, f"expected ~0, got {float(p0)}"

    # perturb exactly one GLOBAL param and one LOCAL param by constant c
    c = 0.5
    pg_name = "ds_net.first_conv.weight"     # global
    pl_name = "dn_resnet.conv1.weight"       # local
    params = dict(m.named_parameters())
    with torch.no_grad():
        params[pg_name].add_(c)
        params[pl_name].add_(c)
    ng = params[pg_name].numel()
    nl = params[pl_name].numel()

    lam = {"global": 3.0, "local": 0.2, "fusion": 0.2}
    pen = float(F.proximal_penalty(m, ref, lam))
    expected = (lam["global"] / 2) * (c ** 2) * ng + (lam["local"] / 2) * (c ** 2) * nl
    print(f"[proximal] penalty={pen:.4f} expected={expected:.4f} (ng={ng}, nl={nl})")
    assert abs(pen - expected) / expected < 1e-5, (pen, expected)

    # scalar-lambda path matches a uniform dict
    pen_scalar = float(F.proximal_penalty(m, ref, 1.0))
    pen_uniform = float(F.proximal_penalty(m, ref, {"global": 1.0, "local": 1.0, "fusion": 1.0}))
    assert abs(pen_scalar - pen_uniform) / pen_uniform < 1e-6
    print("[proximal] OK")


def test_pretrained_load():
    if not os.path.isfile(CKPT):
        print(f"[pretrained] SKIP (ckpt not found: {CKPT})")
        return
    m = build_model()
    rep = F.tolerant_load_pretrained(m, CKPT, device="cpu")
    own = len(m.state_dict())
    print(f"[pretrained] matched={rep['matched']}/{own} missing={rep['missing_keys']} "
          f"unexpected={rep['unexpected_keys']}")
    assert rep["matched"] / own >= 0.99, "substantial fraction of weights unmatched"
    assert rep.get("backbone_match_fraction", 1.0) >= 0.99
    print("[pretrained] OK")


def test_forward_smoke():
    m = build_model().eval()
    # full-size input is required: global net downsamples by 64 -> cam (46,30)
    x = torch.randn(1, 1, 2944, 1920)
    with torch.no_grad():
        out = m(x)
    # native GMIC: forward returns (fusion_logits, y_global, y_local, saliency)
    assert isinstance(out, tuple) and len(out) == 4, type(out)
    fusion_logits, y_global, y_local, saliency = out
    assert fusion_logits.shape == (1, 2), fusion_logits.shape
    assert y_global.shape == (1, 2) and y_local.shape == (1, 2)
    assert saliency.dim() == 4 and saliency.shape[1] == 2, saliency.shape
    # global/local/saliency are probabilities in [0,1]; fusion is a raw logit
    for name, p in [("y_global", y_global), ("y_local", y_local), ("saliency", saliency)]:
        assert float(p.min()) >= -1e-6 and float(p.max()) <= 1 + 1e-6, (name, float(p.min()), float(p.max()))
    s = F.malignant_score(out)
    assert s.shape == (1,) and 0.0 <= float(s) <= 1.0
    print(f"[forward] 4-tuple OK; fusion={tuple(fusion_logits.shape)} saliency={tuple(saliency.shape)} mal_score={float(s):.3f}")


def test_gmic_loss():
    m = build_model().train()
    x = torch.randn(2, 1, 2944, 1920)
    t = torch.tensor([0, 1])
    out = m(x)
    # pos_weight is shared across heads; changing it changes the loss
    l_pw1 = float(F.gmic_malignant_loss(out, t, lambda_l1=1e-5, pos_weight=1.0))
    l_pw10 = float(F.gmic_malignant_loss(out, t, lambda_l1=1e-5, pos_weight=10.0))
    assert l_pw1 != l_pw10, (l_pw1, l_pw10)
    # main loss is finite/positive and backprops into the malignant fusion head
    loss = F.gmic_malignant_loss(out, t, lambda_l1=1e-5, pos_weight=3.0)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    g = dict(m.named_parameters())["fusion_dnn.weight"].grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    print(f"[gmic_loss] OK loss={float(loss):.4f} (pw1={l_pw1:.4f} pw10={l_pw10:.4f})")


def test_ditto_persistence():
    """v is initialized once and continues across rounds (never reset to global)."""
    w = build_model()
    v = copy.deepcopy(w)                       # init v once from (pretrained) w
    v_opt = torch.optim.SGD(v.parameters(), lr=0.1)
    pname = "fusion_dnn.weight"
    v_before = dict(v.named_parameters())[pname].detach().clone()

    # round 1: a personal-pass step moves v
    loss = (dict(v.named_parameters())[pname] ** 2).sum()
    v_opt.zero_grad(); loss.backward(); v_opt.step()
    v_after_r1 = dict(v.named_parameters())[pname].detach().clone()
    assert not torch.allclose(v_before, v_after_r1), "v did not update"

    # round 2: new global arrives into w; v must NOT be reset to it
    new_global = build_model()
    F.load_global_into(w, new_global.state_dict())   # w tracks global
    v_after_r2_start = dict(v.named_parameters())[pname].detach().clone()
    assert torch.allclose(v_after_r1, v_after_r2_start), "v was reset to global!"
    print("[ditto] v persists across rounds and is not reset to global. OK")


def test_amp_loss_boundary():
    """B1 regression on the bf16 boundary, model-free.

    Mirrors the use_amp path: on autocast-produced bf16 outputs, both the loss (its BCE
    terms need fp32) and malignant_score (`.numpy()` on bf16 raises) must work. This is
    the exact path that crashed the use_amp=True executor run.
    """
    B = 4
    t = torch.tensor([0, 1, 0, 1])
    out = (
        torch.randn(B, 2, dtype=torch.bfloat16, requires_grad=True),   # fusion logits
        torch.rand(B, 2, dtype=torch.bfloat16, requires_grad=True),    # y_global prob
        torch.rand(B, 2, dtype=torch.bfloat16, requires_grad=True),    # y_local prob
        torch.rand(B, 2, 46, 30, dtype=torch.bfloat16, requires_grad=True),  # saliency
    )
    loss = F.gmic_malignant_loss(out, t, pos_weight=3.0)
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    sn = F.malignant_score(out).detach().cpu().numpy()   # bf16 -> must cast to fp32
    assert sn.shape == (B,) and float(sn.min()) >= 0.0 and float(sn.max()) <= 1.0
    print(f"[amp] bf16 loss + malignant_score().numpy() OK (loss={float(loss):.4f})")


ALL = [
    test_grouping,
    test_bn_skip,
    test_proximal,
    test_pretrained_load,
    test_forward_smoke,
    test_gmic_loss,
    test_amp_loss_boundary,
    test_ditto_persistence,
]

if __name__ == "__main__":
    failures = 0
    for t in ALL:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
    print("\n==== %d/%d passed ====" % (len(ALL) - failures, len(ALL)))
    sys.exit(1 if failures else 0)
