# Resuming the module-wise Ditto run after a crash

HPU's 16 GiB A4000 can't hold the fp32 personal pass (it overflows to ~22 GiB, oversubscribes to
host RAM, and thrashes at ~2.3 hr/round), so the run progresses slowly and dies every ~20 rounds
(as `1dc61297` did). That's accepted — the plan is crash → resume → repeat until 60 rounds finish.
This is the procedure. It is safe to repeat any number of times; round numbering stays contiguous.

## Why it resumes cleanly

A client returns its trained global to the server only at the END of `execute()` — after the
personal pass and the `{site}_gmic_model_round_N.pth` save. So the server aggregates round N (and
advances to N+1) only once **every** site has finished round N and written both:

- `global_trajectory/{site}_global_round_N.pth` — the sent global `w^N`, reconstructed by the
  reseed (Ditto reads this, NOT the deployed ckpt, because the deployed model is the personal `v`).
- `{site}_gmic_model_round_N.pth` — the deployed personal model `v` at round N, reloaded so `v`
  continues instead of cold-starting.

Requires `cache_global_trajectory: true` (set) and an `output_dir` on persistent storage (it is —
that's how the original round-20 resume worked).

## Steps

1. **Find N** — the last round completed by ALL sites. Run the helper against the run's results dir:

   ```
   python app/custom/resume_params.py \
       /workspace/data/processed/ditto_mw_g0.01_l0.5_f0_amp_3client_20260721
   ```

   It prints `resume_from_local_round` and the server `num_rounds`. (Manual equivalent: the highest
   `{site}_gmic_model_round_N.pth` present for HPU — HPU is the crasher, so its highest is the safe
   minimum.)

2. **Edit `app/config/config_fed_client.json`:**
   - `resume_from_local_round: N`
   - leave `output_dir` and `resume_ckpt_dir` unchanged (the reseed reads the round-N ckpts there).
   - leave `personal_amp_by_site`, `use_amp`, etc. as-is (precision/per-site are preserved).

3. **Edit `app/config/config_fed_server.json`:**
   - `num_rounds: 60 - N`  (raw round 0 reseeds round N; raw 1..(59-N) train logical N+1..59, landing
     the final-round test at logical 59 = the original 60-round plan).

4. **Resubmit** the job. At raw round 0 each client re-submits its cached round-N weights UNTRAINED;
   the server's weighted aggregation reconstructs the round-N global, `v` is restored from its
   round-N ckpt, and training continues at logical N+1. Confirm in the log:
   - `[resume] loaded HPU round-N weights ... submitting UNTRAINED for server re-aggregation`
   - `[resume] restored personal model v <- ...`  (the `v` optimizer's Adam moments are rebuilt
     fresh — never persisted — and re-warm within a few steps; this is the only lossy part.)

5. When it dies again, repeat from step 1. Each resume finds the new (higher) N from the files the
   previous segment saved, so numbering stays contiguous to logical 59.

## Notes

- If a site shows `personal ckpts without a global_trajectory match`, that round can't be reseeded
  for it — the helper already excludes it and picks a lower safe N.
- After the FINAL segment reaches logical 59, run `pool_report_job` (repointed at this results dir,
  with `ditto_modulewise_perround` in the method list) for the pooled personal-model endpoint.
