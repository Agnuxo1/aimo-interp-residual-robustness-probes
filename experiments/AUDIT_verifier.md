# AUDIT verifier - AIMO-Interp Small Track (read-only, 2026-10-28)
Method: read code + reran CV on CPU with features-local/*.npz (scripts in scratchpad, nothing in the project modified).
Reproduction: analysis.oof(last:band+mean:0.75+mean:1, C=0.1, 5 seeds) -> bacc 0.756 / AUC 0.825 / n=97 (matches ensembles.csv 0.7562/0.825). Reported numbers are reproducible.

## Ranking by severity
1. ISSUE (HIGH) Duplicate problems with different whitespace leak across folds. build_dataset.py:17-18 pid = sha1(strip(text)); only outer strip, so CRLF vs LF
   variants get distinct ids. 8 groups of twins in data/problems.jsonl (186 ids, 8 twin pairs after \s+ removal; e.g. c394bacf0413/cd15d0c0441c, 4a0fb5aea663/bd512df7db18,
   7a4a5a78f3ab/814d01f76f35, 647ed8935879/a1b89c29dc4b, 7f1ed9a1aa42/1db461870de6, 308a2b311dd7/66ce9de758ec, 381c5826f189/d746bbceaacc, 7a2660238b10/f952eeb78b2a).
   cv_probe_lib.py:35 excludes proxies only by problem_id, and folds group by problem_id (cv_probe.py:53-55, analysis.py:56-57), so the twin of a TEST row
   can be a TRAIN proxy row (near-identical features + other models' label). 8 of 97 eval rows are affected (DeepSeek 5, Skywork 2, Qwen 1).
   Effect measured (same code, folds grouped by whitespace-normalised text): bacc 0.756->0.727, AUC 0.825->0.821, olympiad bacc 0.635->0.586, olympiad AUC 0.762->0.727,
   per-model AUC Qwen 0.88->0.81, Olmo 0.90->0.86. Headline is optimistic by ~0.03 bacc; olympiad subset by ~0.05. Fix: pid on re.sub(r"\s+"," ") text (or merge twins) and rebuild.
2. ISSUE (HIGH) Headline metric pooled over 4 models/2 sources is easier than the likely target. Own eval rows: DeepSeek 64/97 (66%), 54 of them from
   augmented-sample (80% spurious); base rates per model differ (Qwen .67, DS .28, Olmo .78, Skywork .75). Pooled AUC/bacc partly measures model/source identity.
   Olympiad-only (train-main-v2): n=43 rows but only 22 unique problems (same problems labelled for all 4 models), AUC 0.774, 95% problem-bootstrap CI [0.55, 0.95];
   Skywork AUC 0.556 (~chance), DeepSeek 0.714. A predictor using ONLY the per-model base rate (no features) gets pooled AUC 0.71 / bacc 0.704 (probe: 0.825 / 0.756; the 'length' control 0.70 for the same reason); on olympiad-only the model-prior AUC is 0.548, so the true feature gain is best judged there. Report olympiad-only / per-model numbers as the main figure.
3. ISSUE (MEDIUM) Selection bias / no nested CV. The final config (last:band+mean:0.75+mean:1, C=0.1) was chosen from ~54 (cv_results) + 22 (ensembles) + 42 (layer sweep)
   configurations scored on the same OOF rows (n=97, 22 olympiad problems). Its edge over alternatives is within noise (last:band+mean:1: AUC 0.825; selected 0.825; last:band alone 0.807).
   Paired bootstrap vs controls (bootstrap_ci.json): AUC delta vs length +0.095 [-0.014,0.204], vs tfidf +0.104 [-0.006,0.214], vs mean:0 +0.084 [-0.012,0.187]; ALL CIs include 0
   -> 'beats controls' is not statistically supported. Simulated: with shuffled labels, best-of-12 candidate configs gives bacc 0.559 on average (max 0.655 in 12 reps; avg config 0.488), so selection alone explains ~+0.06 over chance, not the observed 0.756; the optimism is real but moderate.
4. ISSUE (LOW/doc) Missing evidence artifacts: run_all_analyses.sh/run_more_analyses.sh run perm, weights and --eval-all, but experiments/ has no permutation_null.csv,
   permutation_observed.json, own_weight.csv, bootstrap_ci_all.json, oof_selected_all.csv. Any permutation p-value claim is NO COMPROBABLE from the repo.
5. ISSUE (LOW) Proxy rounding (cv_probe_lib.py:36): pandas .round() = numpy half-to-even, so mean==0.5 -> 0.0 (spurious). Exactly 2 tied problems per target model
   (ties are not documented). Switching ties to robust changes bacc 0.756->0.735, AUC 0.825->0.823 (noise level, but the choice is arbitrary and is a hidden bias to 'spurious').
   Proxy means also weight duplicated rows (val-sample and challenge-sample-full repeat 24-28 (problem,model) pairs; no label conflicts) and give 'qwen3-8b:low' 148 rows
   vs 13-64 for other models; the majority is by row count, not by labelling model.
6. ISSUE (LOW) Duplicate own rows count several times: DeepSeek problem 9d3080b91a55 appears 3x, 62 unique pids for 64 rows (bootstrap oof_selected.csv has 95 rows because
   groupby(model,pid,source,y) collapses equal rows; 97 in cv). Stays inside one fold (no leak) but inflates variance weights.
7. ISSUE (LOW, unverifiable) Features were stored float16 (extract.py:30-31) and cast to float32 for training, while inference uses fresh float32 hidden states
   (aimo_features.py:103-104). No inf/NaN in stored features (max|x| 2596), rounding error ~1e-3 relative; no GPU run here -> NO COMPROBABLE end-to-end effect.
   solution.py:52-53 swallows every exception (e.g. CUDA OOM on a long problem) and returns True silently; also fallback True when artifact missing (solution.py:43-44).

## Per-request checks
1. Leakage across folds: ISSUE. By id: no leak (fold built from np.unique(groups) per target; proxies exclude target-owned pids, cv_probe_lib.py:35; proxy never duplicates a test pid).
   By text: see item 1 (8 twin pairs, CRLF vs LF; prefix-100 normalised check finds 12 collisions, the extra 4 are near-duplicates not exact twins, not exploited further).
   Cross-model leakage: PASS (each target fitted separately; proxies from other models only on problems where target has no own label).
2. Preprocessing on test data: scaler and PCA are inside make_pipeline and fit on train folds only (cv_probe.py:36-40, analysis.py:39-43): PASS.
   Band normalisation (cv_probe_lib.py:52) uses mean/std over ALL rows incl. test: transductive leak by construction, but measured effect is ~0:
   recomputing band stats on train rows only gives identical bacc/AUC/oly metrics (0.756/0.825/0.635/0.762 both ways; max |dp| 0.0011, 0 prediction flips) -> ISSUE formal, impact negligible. Fix: fit mu/sd per fold.
3. Labels: thresholds PASS. build_dataset.py:21-26 robust<=0.10, spurious>=0.25, else None; 25 of 393 rows NaN (dropped at cv_probe_lib.py:18), no float-edge errors
   (7 rows at exactly 0.10 -> robust, 6 at 0.25 -> spurious). Derived vs official label agreement 1.000 on 162 rows; official NaN exactly where derived NaN.
   Own class balance: 55 spurious / 42 robust (train-main-v2 12/31; augmented 43/11); proxies are 31% robust; class_weight='balanced' compensates in training.
   Majority rounding: ISSUE minor (item 5).
4. Train/inference consistency: PASS (numerically verified). Loaded submission/probe_artifacts/*.npz, ran solution._probe_prob on stored features and compared with an independent
   refit through cv_probe_lib.featurize (kinds last:band+mean:0.75+mean:1, C=0.1, PCA16): max |dp| = 5e-6 (Qwen), 1e-6 (others). Same layer index int(float(layer)*L) with L=n_layers_plus1-1,
   same band slice lo..hi inclusive, scaler -> PCA -> LR order, threshold 0.5 on mean probability (as CV ensemble). Band mu/sd are fixed training-set statistics at inference (CV used
   all-rows stats; impact ~0 per check 2). are_robust returns native bool via bool(...), order-preserving, [] for empty input. server/aimo_features.py == submission/aimo_features.py;
   e2e/solution.py == submission/solution.py. In-sample own accuracy of the final bundle: 0.75/0.81/0.89/0.75 (not a performance estimate).
   Note: reasoning_effort argument is ignored; CV pooled all efforts per base model.
5. Optimistic metrics: yes, items 1-3 (leak ~ -0.03 bacc, pooled-model/source mix, selection on the same OOF). Nothing found that makes CV numerically wrong (no label leakage in features,
   no test-fit scaler/PCA, seeds average over fold assignments, metrics only on own rows).

## Bottom line
Honest estimate: pooled bacc ~0.73 / AUC ~0.82 after de-duplication; olympiad-only bacc ~0.59-0.63 / AUC ~0.73-0.77 with a CI of roughly +-0.2 (22 problems), and no control is significantly beaten.
Packaging is consistent with CV. Recommended before submission: dedupe twin ids, per-fold band stats, report olympiad-only + per-model metrics, nested/held-out selection, document tie rule.
