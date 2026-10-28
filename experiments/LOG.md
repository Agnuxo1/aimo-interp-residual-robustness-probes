# Experiment log — AIMO-Interp Small Models Track

Protocol: hypothesis → design → control → run with fixed seeds → official metric (accuracy) plus balanced accuracy/AUC (test set is near-balanced) → accept/reject. Evaluation population = rows labelled by one of the four Small-Track models; splits grouped by problem (no problem in both train and test); 5 seeds x 5 folds unless stated. Negative results are logged.

## E0 — contract check (2026-10-28)
- Hypothesis: the always-true bundle passes the official ingestion contract for the Small track.
- Local run (official run_local.py, public val-sample, --small): accuracy 0.375, coverage 1.0, invalid 0.
- Codabench submission 1015190 (solution-always-true-small.zip) uploaded 2026-10-28 08:26 CET.
- Result: Codabench 1015190 Finished, Small Model Track, accuracy 0.50, coverage 1.0, invalid 0 (public 14-case set). Contract accepted.

## E1 — layer/pooling grid (2026-10-28, job f2260c0d9ed842d6 features; CPU CV on servidor-ia)
- Hypothesis H1: robustness is linearly decodable from the evaluated model's residual stream before generation, beyond text-surface controls.
- Design: per model, StandardScaler -> PCA(16) -> logistic regression (class_weight=balanced), C in {0.01, 0.1}; features = last prompt token or token mean at relative depth {0, .25, .5, .75, 1} or the standardised 0.4L-0.8L band. "pooled": add problem-level proxy labels from other labelling models (only on problems without an own label). Evaluation only on own-model labels (n=97: 43 olympiad rows from train-main-v2 + 54 MATH rows for DeepSeek). 5 seeds x 5 grouped folds.
- Controls: length features (4 surface statistics), TF-IDF logistic, layer-0 (embeddings), shuffled labels (E3).
- Result (acc / bacc / AUROC, mean over seeds): length 0.703/0.699/0.749; TF-IDF 0.676/0.680/0.731; embeddings mean-pool 0.660/0.662/0.744; last:0.5 pooled 0.736/0.752/0.816; last:band pooled 0.742/0.759/0.807. Own-labels-only probes are weaker (best AUROC 0.786) -> pooling labels across labelling models helps.
- Decision: accept H1 provisionally (mid-depth residual stream > all surface controls on AUROC); confirm with permutation test and paired bootstrap.

## E2 — probe ensembles (same protocol)
- Hypothesis: averaging probes from complementary readouts (last-token band + late mean-pool) improves stability.
- Result: last:band+mean:0.75+mean:1, C=0.1: acc 0.748±0.017, bacc 0.756±0.014, AUROC 0.825±0.009; olympiad subset bacc 0.635±0.020, AUROC 0.762±0.040. Adding length features hurts (AUROC 0.791).
- Decision: accepted as submission candidate S1 (selected among 22 configurations: mild selection bias, disclosed).

## E2b — end-to-end contract test of S1 (job b441079c0e034cd8, RTX 3090, official run_local.py)
- 43 own-labelled train-main-v2 cases, all four models: coverage 1.0, invalid 0, identical output in two runs (in-sample accuracy 0.791, NOT a generalisation estimate). Job wall time 430 s for two full runs including cold model loads (~215 s per run).
- Bundle SHA-256 37af28ba9ee197fd60340fbd60e6959add4155778e097fe91c434d7cc16ded9d.
- Codabench submission 1015306 (S1-probe-ensemble-small.zip), 2026-10-28 09:41 CET.

## E3 — layer sweep (C=0.1, pooled labels, 21 relative depths x 2 poolings, 5 seeds)
- Last token: layer 0 = 0.50 AUROC by construction (identical final template token); 0.663 at 5 % depth; 0.78 at 25-30 %; plateau 0.80-0.81 from 40 % to the last layer (max 0.813 at 50 % and 85-90 %).
- Token mean: 0.740 at layer 0 (bag of embeddings) rising to 0.80 at 65-85 % depth.
- Decision: supports H1's depth prediction (signal is built during prompt processing, not present in raw embeddings at the readout position).

## E4 — paired bootstrap over problems (2000 resamples, seed-averaged OOF probabilities)
- S1 minus control, AUROC: length +0.095 [-0.014, +0.204]; TF-IDF +0.104 [-0.006, +0.214]; embeddings +0.084 [-0.012, +0.187].
- Balanced accuracy: length +0.067 [-0.080, +0.193]; TF-IDF +0.031 [-0.086, +0.150]; embeddings +0.081 [-0.050, +0.207].
- Decision: effect sizes are consistently positive (~+0.09-0.10 AUROC) but 95 % intervals include 0 with only 97 own-labelled rows. Not conclusive at 95 % on own labels alone -> E7 adds power with problem-level labels.
- Codabench 1015306 Finished: public 14-case Small set accuracy 0.571 (8/14), coverage 1.0, invalid 0. With 14 cases one case = 0.071; used only as a validity check (JEV decision).

## Pre-fix results kept for the record (superseded by v2 below)
- E3b permutation (200 shuffles, S1 config): p = 0.005 for AUROC and bacc, p = 0.010 olympiad AUROC.
- E5 own-label weight 1/2/4: AUROC 0.825/0.825/0.823 -> no gain, rejected.
- E7 problem-level evaluation (732 OOF rows incl. proxy labels): dAUROC vs length +0.057 [+0.007, +0.108]; vs TF-IDF +0.049 [-0.020, +0.116]; vs embeddings +0.031 [+0.000, +0.062].

## AUDIT (independent cog-verifier, experiments/AUDIT_verifier.md) -> v2 data fix
- ISSUE (high): problem_id hashed text with only strip(); 8 pairs of identical problems differing in CRLF vs LF had different ids, so one twin could sit in train (as proxy row) while the other was tested. Measured effect: bacc 0.756 -> 0.727, olympiad bacc 0.635 -> 0.586.
- Fix (v2): ids hash whitespace-collapsed text; texts normalised to LF in both build_dataset.py and aimo_features.build_prompt (so training and inference see identical prompts); one row per (problem, labelling model) with max_drop averaged across sources; proxy majority counts each labelling model once and drops ties (previously pandas half-to-even rounding sent ties to "spurious"). Result: 178 unique problems, 363 label rows, 95 own-labelled rows over 74 problems (22 olympiad problems).
- Other audit points: band standardisation used all rows (transductive, label-free) -> measured effect max |dp| = 0.001, kept and disclosed; train/inference numerics identical to 5e-6; metrics mix models -> v2 adds a per-model prior control and within-model AUROC; selection on the same CV -> v2 adds nested selection.
- Features re-extracted (job 6a2749a15ac549c0).

## E8 — v2 ensembles and controls (5 seeds x 5 grouped folds, 95 own rows)
| config | acc | bacc | AUROC | within-model AUROC | olympiad bacc | olympiad AUROC |
|---|---|---|---|---|---|---|
| last:band+mean:1, C=0.1 | 0.724 | 0.735±0.024 | 0.812±0.012 | 0.762±0.038 | 0.593 | 0.737 |
| last:band+mean:1+last:0.5, C=0.01 | 0.726 | 0.741±0.022 | 0.820±0.013 | 0.757±0.043 | 0.567 | 0.730 |
| last:0.5, C=0.01 | 0.745 | 0.761±0.016 | 0.810±0.016 | 0.700±0.055 | 0.570 | 0.677 |
| S1 = last:band+mean:0.75+mean:1, C=0.1 | 0.705 | 0.715±0.027 | 0.815±0.015 | 0.750±0.033 | 0.531 | 0.726 |
| control: per-model prior | 0.558 | 0.500 | 0.438 | 0.490 | 0.500 | 0.463 |
| control: surface length, C=0.1 | 0.705 | 0.699 | 0.739 | 0.629 | 0.548 | 0.532 |
| control: TF-IDF, C=0.01 | 0.642 | 0.652 | 0.734 | 0.622 | 0.464 | 0.591 |
| control: embeddings mean-pool, C=0.01 | 0.676 | 0.680 | 0.766 | 0.731 | 0.552 | 0.679 |
- Reading: after the fix S1 drops (olympiad bacc 0.531). last:band+mean:1 is best on the two metrics closest to the test conditions (within-model AUROC and olympiad subset) and second on overall AUROC.
- Nested selection (selection redone inside each outer split; 8 candidates; inner 4-fold): acc 0.720, bacc 0.732±0.023, AUROC 0.805±0.012, within-model AUROC 0.746±0.016, olympiad bacc 0.565±0.041, olympiad AUROC 0.701±0.019. Inner picks vary (last:band 25, last:0.5 21, mean:1 21, ...). Selection optimism is small (~0.01-0.02).
- JEV decision 3 (provenance=jev, conf 0.96): submit A = last:band+mean:1, C=0.1 as S2.

## E6 — causal directional ablation inside the forward pass (job cf7b0d8ad7c446fc, v2 features)
- Hypothesis H2: the robustness information read by final-layer probes is routed through the direction (rank 1) or the 16-dim PCA subspace that a mid-depth (L/2) token-mean probe uses; projecting it out of the residual stream at L/2 for every token should lower the downstream AUROC more than projecting out random directions of equal rank.
- Design: per grouped fold (seed 0), mid probe and downstream probes fitted on training problems only; held-out own-labelled problems re-run with a forward hook; controls = no ablation and 3 random directions/subspaces per rank.
- Result (AUROC of final-layer token-mean / last-token probes): none 0.735/0.757; probe rank-1 0.739/0.760; probe rank-16 0.750/0.757; random rank-1 0.724-0.736/0.758-0.765; random rank-16 0.723-0.735/0.756-0.787. Recomputed vs stored features (no ablation): max abs diff 0.031 (bf16 kernels).
- Decision: H2 rejected. Linear erasure at a single mid layer does not remove downstream decodability: the signal is redundantly encoded and can be re-derived by later layers (consistent with known limits of single-layer linear erasure). Reported as a negative result.

## E9 — v2 statistics for the selected configuration A (last:band+mean:1, C=0.1)
- Permutation (200 label shuffles, seeds 1000-1199): observed AUROC 0.812, within-model 0.762, bacc 0.735, olympiad AUROC 0.737; p = 0.005 for all four (best null AUROC 0.651, within 0.723, bacc 0.638).
- Paired bootstrap, own labels (95 rows / 74 problems), A minus control: length dAUROC +0.086 [-0.031, +0.206], d within +0.183 [-0.097, +0.504]; TF-IDF dAUROC +0.081 [-0.034, +0.185], d bacc +0.128 [0.000, +0.249]; embeddings dAUROC +0.061 [-0.035, +0.163].
- Paired bootstrap, problem-level (own + proxy labels, 700 rows / 175 problems): length dAUROC +0.052 [+0.002, +0.104], d within +0.054 [+0.003, +0.106]; TF-IDF +0.048 [-0.017, +0.111]; embeddings +0.025 [-0.006, +0.058].
- Per-model prior control: its OOF AUROC (0.44) is below 0.5 because leaving a group out shifts the training base rate against the held-out label; within-model AUROC is the base-rate-free metric we rely on.
- Layer sweep v2 (last token): 0.50 @0, 0.744 @10 %, 0.787 @30 %, peak 0.816 @40 %, 0.78-0.81 afterwards.
- Decision: H1 accepted (permutation); white-box advantage over surface statistics significant at problem level, positive but not significant against embeddings.

## S2 — final submission (Codabench 1015694, 2026-10-28 12:51 CET)
- Bundle: dist/S2-probe-A-small.zip, SHA-256 3281b1dfaa4566cdbe7917da75dfaf6d2ac5ddb455c92d74bbb806722a73a2ba (probes exported from v2 features, prompt normalisation included).
- Contract test (job 369db1e321c241b6, official run_local.py --small, 43 cases, 2 runs): coverage 1.0, invalid 0, identical predictions; ~165 s per run on an RTX 3090 including cold loads.
- Codabench: Finished, public 14-case accuracy 0.571, coverage 1.0, invalid 0 (validity check only).
