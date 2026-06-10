# One FedAvg run → Ditto by replay (no FedAvg recompute per λ)

Ditto's global model is trained by plain FedAvg; the personal model `v` is trained with
`task_loss(v) + (λ/2)‖v − w^t‖²`, anchored each round to the start-of-round global `w^t`. The personal
pass never affects the global, so **one FedAvg run's per-round global trajectory is the correct anchor
for every λ**. `replay_personalized` reproduces the interleaved personal trajectory from that cache —
proven bit-identical in `test_ditto_replay_equivalence.py`. So: run FedAvg once, replay Ditto N times.

## Recommended: run FedAvg in the simulator (no HPU)

The Ditto sweeps already run in the **simulator on the DGX** (all three sites' data local). For the
replay to reproduce the interleaved sweep, the FedAvg trajectory must come from that **same simulator,
data, and seed** — which also means it never touches HPU, so there are **no nightly crashes and no
resume to manage**. The HPU-crash/numbering worry only applies to a *real-world* FedAvg run (next
section); for informing Ditto, you don't need it.

```bash
# 1) build the per-site crop caches once (skip if already built by the ditto sweeps)
python ditto_sweep/launcher.py --base-job ../gmic_job --prepare-cache --gpu-pools "0,1,2"

# 2) one FedAvg run that caches the per-round global trajectory
python ditto_sweep/launcher.py --base-job ../gmic_job --fedavg --rounds 60 \
  --gpu-pools "0,1,2" --output-base /workspace/sim/fedavg --work-root /workspace/sim/fedavg_runs
```

This writes, per site, `…/fedavg/<site>/incoming_global/<site>_incoming_global_round_{0..59}.pth` —
the start-of-round global each round (`round_t` = `w^{t-1}`; round 0 = the pretrained seed). The
broadcast global is identical at every site, so **any one site's** cache is the trajectory.

## Then: replay the Ditto sweep against that trajectory

```bash
# scalar λ
python ditto_sweep/replay_sweep.py --base-job ../gmic_job \
  --traj-dir /workspace/sim/fedavg/fedavg/UHCC/incoming_global --traj-prefix UHCC \
  --lambdas 0.05,0.1,0.5 --clients UHCC,HPU,RSNA-GCP --gpu 0 \
  --out /workspace/sim/ditto_replay/sweep_summary.json

# module-wise, anchored at the scalar best, global tethered up / heads freer
python ditto_sweep/replay_sweep.py --base-job ../gmic_job \
  --traj-dir /workspace/sim/fedavg/fedavg/UHCC/incoming_global --traj-prefix UHCC --modulewise \
  --anchor 0.1 --global-values 0.1,0.5,1.0 --local-values 0.01,0.05,0.1 --fusion-values 0.01,0.05,0.1 \
  --clients UHCC,HPU,RSNA-GCP --gpu 0 --out /workspace/sim/ditto_replay/mw_summary.json
```

Each config takes seconds–minutes (one personal fit per site, no federation), versus a full FedAvg per
λ in the interleaved launcher. Ranking is the same worst/mean-site val-AUC selection.

**Sanity check after launch:** each site's round-0 replay val AUC should ≈ its pretrained baseline
(`v` starts at the pretrained init, anchored to itself). And the all-anchor module-wise baseline
(`g=l=f=0.1`) should track the scalar λ=0.1 replay.

## If you DO run FedAvg real-world (gmic_job_hpu, with HPU): surviving nightly crashes

Only needed if the paper requires the actual cross-site federated FedAvg (not the simulator). Round
numbering is now resume-safe: every per-round artifact is stamped `current_round + resume_from_local_round`,
so a resumed segment continues the logical numbering instead of overwriting round 0.

Per crash:
1. Find the last **completed** logical round `N` (highest `…_gmic_model_round_{N}.pth` / trajectory file).
2. In `config_fed_client.json` set `"resume_from_local_round": N` (and `resume_ckpt_dir` → the run dir
   holding those files). Keep `cache_incoming_global: true`.
3. Resubmit the job. Round 0 reseeds (server reconstructs the round-`N` global from each site's cached
   weights, train-size weighted); round 1 = logical `N+1`; trajectory files continue `N+1, N+2, …`.
4. Repeat each crash with `resume_from_local_round` = the new last-completed logical round.

The result is one contiguous `incoming_global_round_{0..M}.pth` trajectory across however many segments
it took — directly usable by `replay_sweep.py`. (Gather one site's `incoming_global/` to the DGX,
since the replay trains per-site `v` and needs the DGX's local site data.)

## Why replay == interleaved (and the one caveat)

`test_ditto_replay_equivalence.py` proves the anchor mapping (`anchor[t]` = start-of-round global) and
v-carried-forward reproduce interleaved Ditto **bit-for-bit** on fixed data. With the real balanced
sampler, replay won't bit-match a live interleaved run (its global pass would have advanced the RNG
first) — that's sampling noise, not bias, and is fine for λ selection on val AUC.
