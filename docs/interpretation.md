# Feature interpretation

`interpret.py` has an LLM write a title and a one- or two-sentence description for each SAE feature, then checks them two ways: on recorded activations the LLM hasn't seen, and on the studied model itself. Below, "name" means the title and description together. Titles, descriptions, scores, and test texts show up in the explorer.

![For each feature: sample recorded activations, name the feature, check the name on held-out tokens, write test texts that the studied model scores, revise a failing name once, and steer the decoder direction](../assets/interpretation_pipeline.svg)

## Run

```sh
llama-server -m gemma-4-31B-it-Q8_0.gguf --port 8080 -c 32768 -np 4 -cram 0 --reasoning off
python3 interpret.py --sae-dir artifacts/qwen3-1.7b-base_l14_w16_k32_500m --base-url http://127.0.0.1:8080/v1 --sample 500
python3 explore.py --sae-dir artifacts/qwen3-1.7b-base_l14_w16_k32_500m
```

- `--sample N` names N features spread evenly over activation density, `--features` names specific ones, and neither names every feature with at least 40 recorded activations.
- Results are appended to `interpretations.jsonl` in the activation directory. A rerun skips features already there, so it resumes where it stopped and retries features that failed.
- `-cram 0` keeps llama-server's prompt cache from filling RAM. `--reasoning off` makes every call answer directly in JSON, which the token limits are sized for. `-np 4` matches the default `--workers 4`.
- On an M4 Max, gemma-4-31B-it (~40GB of RAM) takes about 36 s per feature, 5 hours for 500. gemma-4-12B-it decodes ~3x faster and is a bit less precise ([results](#results)).

## Method

- **Sample and explain**: the explainer sees the 8 strongest activations and 12 drawn evenly from activation-quantile bins (0-50, 50-80, 80-95, 95-100%), 32 tokens before and 8 after, with the firing tokens marked and their activations on a 1-10 scale, plus the tokens the feature fires on most and the share of their occurrences it fires on. Sampling across the range keeps the name from describing only the peak. [1]
- **Detect**: a judge given only the name says whether the feature fires on each of 24 marked tokens it hasn't seen: 12 activations from the same bins, and 12 silent tokens split between random ones, the feature's frequent tokens where it stays silent, and activations of the feature with the closest decoder direction. The last two are hard negatives: a vague or token-level name says yes to them. Balanced accuracy averages recall with the specificity on each kind of silent token. [1]
- **Write tests and measure**: the judge writes 6 pairs of texts from the name, one meant to make the feature fire and a minimal contrast meant to keep it silent, and the model and SAE measure the actual activations; a pair passes when the contrast's activation is under half the first text's (a smaller drop is within what irrelevant changes cause). The model, not a judge, is the ground truth, so this catches names that are right about the examples but too broad. [2, 3]
- **Revise**: when fewer than 4 of 6 pairs pass, the explainer sees the measured activations and revises the name once; the version with the higher detection plus pass rate is kept. [3]
- **Steer**: the decoder direction is added at the SAE's layer on 8 neutral prompts, at the strongest multiple of the max activation (0.5-16x) that keeps the KL from the unsteered next-token predictions within 1 nat, and the tokens whose log-probability rises most are reported. This stays separate from the name, since what a mid-layer feature fires on and what it promotes often differ. [4, 5]
- **Output**: one JSON line per feature with the name, both scores, the measured test pairs, and the steering result. The explorer lists named features first and shows these on each feature page.

## Results

Names come from the Qwen3 1.7B layer-14 run, with gemma-4-31B-it (Q8_0, llama-server) as explainer and judge. The design was settled on 60 features stratified by activation density, then the final pipeline ran on 40 others.

Detection on the 60 features, with the judge given different descriptions (before revision):

| Description | Balanced accuracy |
|---|---:|
| Explainer's | 0.69 |
| The feature's most frequent tokens only | 0.57 |
| Another feature's | 0.51 |

Final pipeline on the 40 other features, in 24 minutes (36 s per feature, 4 parallel requests on an M4 Max Mac Studio):
- Detection balanced accuracy 0.71. The judge says the feature fires on 80% of the activations in the top 5%, 66% at 80-95%, 51% at 50-80% and 28% in the bottom half; and on 3% of random tokens, 14% of the closest feature's activations and 26% of the feature's frequent tokens where it stays silent.
- Generated tests: 63% of pairs pass, 70% of features pass at least 4 of 6, 20% pass none.
- 40% of features pass both checks (detection at least 0.7 and 4 of 6 generated tests). 22% of the names come from the revision.

- Names describe the peak, not the tail: weak activations often land on arbitrary tokens in a topical context (any token of a firearms spec sheet), which a name doesn't predict.
- Detection alone overrates broad names, since the held-out examples share the training examples' quirks. Feature 29056 fires on the comma after page ranges in RePEc citations (`vol. 74(1), pages 119-147, September`): "academic citation volume and page range" scored 1.00 on detection, but none of 6 citations in other formats activated it. The two checks barely correlate (Spearman 0.17): detection measures coverage, generated tests precision.
- Until the margin below was chosen, a pair passed when the first text fired more by any margin; the design-phase numbers here use that rule.
- Revision with the measured activations raised the generated-test pass rate of the 28 failing names from 26% to 44% (10 then passed 4 of 6). Re-testing the original names on freshly sampled texts gave 28% (1 passed), so the gain isn't regression to the mean. Detection held (0.66 to 0.64) as names got narrower: the judge said yes to 41% of same-token negatives instead of 54%.
- Short texts suffice for generated tests: real top-activation excerpts fire as strongly alone as after a preceding paragraph.
- Any margin was too lenient: for feature 4064, an adjective after an indefinite pronoun ("something new", "nothing special"), every contrast still fired at 43-89% of the max, yet "positive or novel qualifiers" passed 5 of 6 pairs. A neutral sentence before a firing text (420 texts, 3 sentences) leaves the median activation unchanged, cuts it below 0.76x for only 5% of texts and halves it for 1.7%, while swapping words the feature doesn't care about moves it by up to half; so a pair now passes only if the contrast stays under half. That rule flips 23 of 175 passing pairs, and 4064 drops to 0 of 6. Agreement between two independent sets of generated texts for the same names falls slightly, from 0.78 to 0.70 (Pearson over features): the stricter rule tests a harder property.
- Logit-lens tokens in the explainer prompt made no difference (detection +0.002 ± 0.020, generated tests -0.025 ± 0.064, 95% CIs) and were dropped.
- gemma-4-12B-it decodes ~3x faster. As judge of the same names its per-feature detection correlates 0.81 with the 31B's, and it is slightly stricter (0.66 vs 0.69 detection, 54% vs 62% generated tests). As explainer, judged by the 12B, its names match the 31B's on detection (0.65 vs 0.66) but pass fewer generated tests (48% vs 54%).
- At a fixed 4x max activation the steering KL spans 0.5 to 5.8 nats (10th to 90th percentile), hence the per-feature calibration; the median feature stays within 1 nat up to 2x. Steering tokens often confirm the name (coal: " coal", " methane", " mines"), but some show an effect the name misses: feature 15393, "major sporting events", steers toward " terrorist", " protests", " ISIS".

<details>
<summary>Command for the 40 features</summary>

```sh
llama-server -m gemma-4-31B-it-Q8_0.gguf --port 8080 -c 32768 -np 4 -cram 0 --reasoning off
python3 interpret.py --sae-dir artifacts/qwen3-1.7b-base_l14_w16_k32_500m --base-url http://127.0.0.1:8080/v1 \
  --features 29735 7213 3573 4023 5255 20162 11626 12119 21637 10036 27736 25036 3303 16911 16739 5989 5509 29444 11997 7825 29895 20163 4064 8481 6118 20468 14934 4979 20010 20963 29412 7464 160 18972 16090 1877 27889 16601 5063 21072
```

</details>

Sources:
- [1] [Automatically Interpreting Millions of Features in Large Language Models](https://arxiv.org/abs/2410.13928)
- [2] [FADE: Why Bad Descriptions Happen to Good Features](https://arxiv.org/abs/2502.16994)
- [3] [SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization](https://arxiv.org/abs/2606.08496)
- [4] [SAEs Are Good for Steering – If You Select the Right Features](https://arxiv.org/abs/2505.20063)
- [5] [Enhancing Automated Interpretability with Output-Centric Feature Descriptions](https://arxiv.org/abs/2501.08319)
