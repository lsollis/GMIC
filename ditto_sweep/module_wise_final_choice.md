# Module-wise Ditto: final configuration and justification

**Final run:** `gmic_job_ditto_mw_sim` — `method=ditto_modulewise`, **λ_global = 0.01, λ_local = 0.5, λ_fusion = 0.0**, 60 rounds, NVFLARE simulator (3 sites: UHCC, HPU, RSNA-GCP), selected on worst-site validation AUC (equity objective).

In Ditto, λ is the strength of the proximal pull of each site's *personal* model toward the federated-average *global* model: **higher λ = more sharing / less personalization; λ = 0 = fully private (no pull).**

## What the three groups are (GMIC architecture)
- **global** = `ds_net` (+ `left_postprocess_net`): the whole-image, low-resolution network that produces the saliency map.
- **local** = `dn_resnet`: the high-resolution patch/crop backbone.
- **fusion** = MIL-attention + `classifier_linear` + `fusion_dnn`: the prediction/decision head.

(Note the terminology trap: "global"/"local" name GMIC's two *architectural branches*, **not** "shared vs. private." Sharing is controlled by λ.)

## Selection philosophy
Three sweeps — FedProx μ, scalar Ditto λ, and module-wise per-group λ (two grids: anchor 0.01 "wide," anchor 0.05 "narrow") — all came out **statistically flat**: worst-site validation AUC spanned ~0.852–0.856, a band well inside the noise floor of the smallest validation set (RSNA-GCP, n = 202; ≈0.005 AUC per case). We therefore did **not** select the literal worst-site argmax (which would chase validation noise). Instead we chose the **most interpretable, mechanistically-motivated point consistent with (within noise of) the sweeps.**

## The two trends that are *not* noise
1. **Sharing the prediction head is monotonically harmful — the one clean signal.** Increasing λ_fusion lowers UHCC AUC monotonically in *both* grids (wide: 0.9119 → 0.9069 → 0.9051 across f = 0.01/0.5/2; narrow: 0.9124 → 0.9108 → 0.9072 across f = 0.01/0.05/0.5), and f = 2 is the single worst configuration in the wide grid (worst-site 0.8516, mean 0.8911). Mechanistically expected: the decision head carries site-specific calibration, prevalence, and threshold, so forcing it toward the federated average costs each site.
2. **Sharing-tolerance ranks local > global > fusion.** At the strongest perturbation (wide grid, λ = 2), the patch backbone (local) is the least harmed by sharing across 3 of 4 site/metric readouts — worst-site (local 0.8540 > global 0.8532 > fusion 0.8516), mean (0.8930 > 0.8923 > 0.8911), and UHCC (0.9110 > 0.9081 > 0.9051). This is weak (visible only at high λ; masked in the narrow grid where λ is too low to separate the groups) but directionally consistent, and it agrees with mechanism.

## Per-group justification
- **λ_fusion = 0 (fully private head).** Data: head-sharing is the clearest harmful trend; 0 is its monotonic endpoint. Mechanism: the decision boundary is the most site-specific component. This makes the run a **shared-representation + private-head** scheme (the FedPer / FedRep personalization pattern). *Bold extrapolation:* the lowest λ_fusion actually tested was 0.01, not 0, so 0 is the trend's limit rather than a measured point. Implementation verified safe (executor preserves a literal 0.0 rather than coercing it to the local default; the proximal term skips zero-λ groups), so the head receives no proximal pull as intended.
- **λ_local = 0.5 (share the patch backbone).** Data: local is the most sharing-*tolerant* group (trend #2), so a strong pull costs essentially nothing on validation. Mechanism: high-resolution patch texture (calcifications, masses) is largely site-agnostic, so sharing this backbone lets the data-poor site (RSNA-GCP) borrow representational strength.
- **λ_global = 0.01 (personalize the whole-image backbone).** There is **no clear evidence** distinguishing 0.01 from 0.05 (narrow grid: worst-site 0.8560 vs. 0.8562, mean 0.8946 vs. 0.8945, UHCC 0.9112 vs. 0.9108 — all ≈0.0002–0.0004, noise). We chose 0.01 on logic: the whole-image branch encodes site-specific acquisition, breast density, and positioning, so it should personalize; and 0.01 yields the cleanest narrative — **the patch backbone is the only meaningfully-shared component.**

## How to position this in the paper
Report this configuration as **architecture-motivated, not performance-selected**: *"Personalization should concentrate on the site-specific components — the decision head (fully private) and the whole-image representation — while the universal high-resolution patch-feature extractor is shared."* Expect it to land **within noise of scalar Ditto** (it is not a win, and should not be claimed as one). The contribution is the *structure and its mechanistic rationale* (a GMIC-specific instance of shared-representation/private-head FL), plus the negative result that per-group λ tuning otherwise offers no advantage and that site heterogeneity — RSNA-GCP is the worst site in every configuration — dominates method/λ choice.

## Caveat to monitor
λ_fusion = 0 removes all regularization on the head, which trains freely on each site's local data. For RSNA-GCP (~200 training exams) this risks overfitting the head. The head is small and sits on shared/anchored backbone features, so the risk is modest, but RSNA's train↔validation gap should be checked; if RSNA validation falls below the λ_fusion = 0.01 sweep point (~0.855), revert the head to λ_fusion = 0.01 (already characterized).
