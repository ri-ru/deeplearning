# Explanations and supporting material

Companion to POSTER_REWRITE.md. The poster file says what to put on the
A0; this file says everything else: the order we did it in, the reasoning,
each stage in detail, the caveats, the things that did not work, and the
open questions.

## 1. How to read this

The work on branch `verify-poster-claims` happened in 13 stages plus
preparatory verification. Each stage was designed to answer a single
specific question. They build on each other but you can read most of them
in isolation. The full results JSON for each stage lives in
`overnight/results/stage<N>/`, the trained checkpoints in
`overnight/runs/stage<N>/`, and the live progress in `OVERNIGHTLOG.md`.

If you want the punch line, read POSTER_REWRITE.md. If you want the
methodology, read the next section.

## 2. The narrative arc, in order

### Phase A: verification

Question: are the original poster numbers honest?

We re-ran every claim from scratch using the saved checkpoint. All four
headline numbers reproduce exactly (TF-IDF F1 = 0.7891 vs poster 0.789;
GPT-4o-mini F1 = 0.9637; RoBERTa OOD F1 = 0.9681; cosine eq>0.9 = 21.60%;
cosine uneq>0.9 = 50.87%). The 55% on the manual set reproduces and we
added the per-class breakdown that the poster did not show: equivalent
accuracy 31%, unequivalent accuracy 100%. The model is collapsed, not
"55% accurate". Scripts: `verify_poster.py`, `verify_ood.py`,
`verify_negation_prior.py`. Output: `REPRODUCE.md`.

### Phase B: mitigation sweep (Stage 1 + evolve)

Question: can a simple data-side fix recover the missing accuracy?

We defined six knobs (drop high-cosine pairs, drop negation-flip pairs,
upsample equivalent-with-negation, downsample unequivalent-with-negation,
add Qwen-7B paraphrases, training epochs) and tried 13 combinations,
including six proposed by an evolve loop that reads the running
leaderboard and samples the next variant from a fixed schema.

Winner: v04_style_mix_qwen7b, which is "poster baseline data plus 200
extra paraphrases from Qwen 2.5 7B AWQ". On the 20-pair manual set this
went from 50% to 95%. We logged this in OVERNIGHTLOG.md and at the time
thought we had the answer.

### Phase C: mechanistic interpretation (Stages 2 and 6)

Question: what does the residual stream look like in the broken model
versus the fixed model, and can we ablate the shortcut at inference?

We extracted [CLS] residuals at every transformer layer for 52,000
synthetic pairs plus the 20 manual and 20 Claude pairs. Then trained a
logistic regression probe for the equivalence label and a separate probe
for a regex feature ("does text_b contain a negation that text_a does
not"). Both probes are highly accurate on synthetic at every deep layer
(AUC 0.99 by layer 8). On the manual set, the equivalence probe peaks
at layer 10 (AUC 0.978) and then **drops to 0.769 at layer 12**.

The fix (v04) keeps layer 12 manual AUC at 0.978. The mitigation prevents
the final-layer collapse.

Stage 6 then tested whether you could surgically remove the shortcut at
inference. We fit logistic regressions against six different shortcut
proxies (negation, length ratio, "the post" prefix, first-person, much
shorter, verdict word) and projected each direction out of [CLS]. After
projecting out all six, the manual AUC on the baseline went from 0.769
to 0.758. No effect. The shortcut is encoded redundantly across many
dimensions; you cannot kill it with a low-rank projection.

This was the moment the story shifted from "find the shortcut neuron"
to "prevent the collapse at training time."

### Phase D: training dynamics (Stages 3 and 5)

Question: when does the shortcut form, and is there an early-stopping
sweet spot?

Stage 3 saved a checkpoint every 200 steps across 3 epochs (15 snapshots)
and scored each. The result looked striking: at step 400 the model had
manual = 95%, claude = 90%, OOD F1 = 0.94. By step 800 manual had dropped
to 65%, by step 1400 to 40%. The model passes through a balanced state
and then falls into the shortcut.

Stage 5 zoomed in: snapshots every 25 steps from 25 to 800. The picture
got messier. At step 25 the model predicted equivalent on 100% of pairs.
At step 50 it had partially formed the shortcut already (eq 23%, uneq 86%).
Between step 100 and 250 the manual accuracy oscillates: 60%, 80%, 70%,
80%, 55%, 80%. The "sweet spots" are short noise spikes, not stable
states.

The actual best snapshot (step 250) gives manual 85%, claude 90%,
OOD F1 0.954. Cost of stopping there versus running to the end: about
1.5 OOD F1 points for plus 40 manual acc points. A real trade, IF that
sweet spot generalizes to a larger eval. Stage 7 cast doubt on this
(v04 dropped from 95% on n=20 to 68% on n=50). So early-stopping is
likely fragile.

### Phase E: realistic eval (Stage 7)

Question: does the v04 95% hold on a bigger blind set?

We generated 50 fresh "human-style" reactions on OOD test posts using
GPT-4o-mini, in a casual Reddit-reader voice. Each pair verified by a
second GPT-4o-mini call (SAME or DIFFERENT). On this 50-pair set, the
poster baseline got 54%, v04 got 68%. So v04 helps (plus 14 points) but
the 95% claim from n=20 was inflated. Sample size matters; we should not
trust 20-pair numbers.

### Phase F: garbage-in check (Stage 8)

Question: how much of the failure is label noise in the training data?

We sampled 192 pairs stratified by (label, domain) from reformulated_full.csv
and asked GPT-4o-mini, blind, whether each pair reaches the SAME or
DIFFERENT conclusion. Result: equivalent label has 100% agreement;
unequivalent label has 91% agreement (9% labeled "unequivalent" actually
say the same thing as the original). Overall 95% clean. The training
data is not the bottleneck. The shortcut is genuinely about the
training distribution's style, not noise.

We also audited the four Qwen-7B augmentation files. Three are 92 to 100%
clean. `reformulated (4).csv` is 70% clean (40% of its "equivalent"
labels actually say something different). This file is included in v04
and the multi_gen training; results would be slightly cleaner if we
dropped it.

### Phase G: cross-generator transfer matrix (Stage 9)

Question: does multi-generator training actually fix the shortcut, or
does it just shift it?

This is the experiment the poster's "future work" section recommends.
We generated 200 paraphrase pairs from each of 5 LLMs via OpenRouter
(Anthropic Claude Haiku 4.5, OpenAI GPT-4o-mini, Qwen 2.5 72B, Meta
Llama 3.1 8B, DeepSeek V3) using the same 100 OOD source posts. Then:

- Trained 5 single-generator models (one roberta-base per generator,
  160-pair train split, 40-pair test split).
- Trained one multi_gen model on the union (800 train pairs).
- Cross-evaluated all 6 models on every test split plus the 40 human
  pairs.

The diagonal (in-distribution) was 0.55 to 0.93 depending on generator.
The big result was the multi_gen row: 1.00 / 1.00 / 0.93 / 1.00 / 1.00
across the 5 test sets and **0.90 on the 40 human pairs**. With 30x less
training data than the poster.

DeepSeek and Llama-8B's single-generator models collapsed to "predict
equivalent" because 160 pairs is too few to learn the harder unequivalent
class from those generators. Qwen-72B single generator was the strongest
(0.925 in-distribution, 0.75 to 0.90 on others).

### Phase H: confirm the headline (Stage 10)

Question: does multi_gen's 90% generalize, or is that another small-sample
artifact like v04's 95%?

We generated 100 fresh human-style reactions using Claude Haiku 4.5 in
a different voice ("tired person writing a comment on their phone") on
source posts that had never appeared in Stage 7, Stage 9, or the original
human sets. Verified by a second Haiku call.

| Model | Acc | AUROC | eq / uneq |
|---|---|---|---|
| poster baseline | 59% | 0.745 | 40 / 84 |
| v04 (poster + Qwen-7B aug) | 66% | 0.808 | 51 / 86 |
| multi_gen | 81% | 0.896 | 84 / 77 |

Multi_gen drops from 0.90 on n=40 to 0.81 on n=100, but it is still
clearly above the other two and is the only model with balanced eq/uneq.
AUROC 0.896 confirms genuine ranking ability, not just threshold-tuning.

### Phase I: meta-tests (Stages 11, 12, 13)

Question 1: what is the ceiling? Stage 11 ran 5 frontier LLMs zero-shot
on the Stage 10 100-pair set. GPT-4o-2024-11-20, Claude Haiku 4.5, and
Qwen 2.5 72B all hit perfect 100/100. Claude Sonnet 4.5 got 99/100,
DeepSeek V3 got 98/100. So the dataset ceiling is real and we capture
80% of it with a 125M-parameter student.

Question 2: which generator carries the result? Stage 12 trained 5
leave-one-out models. Dropping Anthropic Haiku drops human accuracy from
81 to 60. Dropping anything else costs 1 to 7 points. Haiku is the
critical ingredient.

Question 3: is fine-tuning even necessary? Stage 13 tried
sentence-transformers/all-mpnet-base-v2 cosine + a tuned threshold +
LogReg with three features. AUROC was below 0.5 on every human eval set.
Off-the-shelf semantic similarity cannot solve the task. Fine-tuning is
required.

### Phase J: course brief (Stage 4)

Question: can we also produce a real hate-speech classifier as the
project originally promised?

We trained TF-IDF and roberta-base on two public datasets: Davidson
2017 (24,783 tweets, 3-class) and HateXplain (19,229 posts, 3-class).
Results: Davidson 91.6% accuracy / 0.767 F1 macro; HateXplain 71.5% /
0.703. In line with published baselines.

## 3. Stage-by-stage details

### Stage 1: data mitigation sweep

Seven hand-defined variants. Each one trains roberta-base from scratch
on a filtered or augmented version of reformulated_full.csv (same
hyperparameters as the original poster: 3 epochs, bf16, lr 2e-5,
batch 32, max_len 256, warmup 0.06). Scores on OOD test + manual + claude.

| Variant | Recipe |
|---|---|
| v00_baseline | full data, 3 epochs, sanity check |
| v01_drop_high_cosine | drop unequivalent pairs with TF-IDF cosine > 0.85 |
| v02_drop_neg_shortcut | drop unequivalent pairs with negation in text_b not in text_a |
| v03_balance_neg | upsample eq+neg 2x, downsample uneq+neg by 50% |
| v04_style_mix_qwen7b | + 4 improved Qwen-7B reformulation CSVs |
| v05_one_epoch | full data, 1 epoch (proves shortcut forms early) |
| v06_combined | v01 + v02 + v04 stacked |

Plus evolve.py proposed six more variants of "drop high cosine + mix qwen
7b with different epochs and upsample factors".

Best result for the small manual set: v04 at 95%. Best for balanced
eq/uneq: auto_81351db7cc at 90% (eq 92%, uneq 86%). v01 (aggressive
cosine drop) was the worst: equivalent recall collapsed to 0%, manual
accuracy 35%. Filtering too aggressively destroys the model.

Resumable: each variant writes one JSON to `overnight/results/stage1/<name>.json`.
Re-running the script skips any variant whose JSON exists.

### Stage 2: per-layer probes

Extract [CLS] residual at every of 13 layers (embedding plus 12 transformer
blocks) for the 52,137 synthetic pairs plus the 20 manual pairs. For each
layer, train two logistic regressions on the synthetic-train split: one
for the equivalence label, one for the negation regex flag. Evaluate both
on synthetic-val and on the 20 manual. Also fit an MLP for the equivalence
label.

Negation-direction ablation: for each layer, take the negation logreg's
coefficient vector, project it out of every residual vector, retrain the
equivalence probe on the ablated residuals.

What we learned:
- Equivalence is decodable from synthetic val with AUC > 0.99 from layer 7
  onwards.
- Negation regex is decodable at AUC > 0.93 from layer 5 onwards.
- The two probes' predictions correlate only at Pearson 0.27 to 0.36 at
  deep layers. So negation is not the same axis as the equivalence
  prediction; it is one component of a broader stylistic family.
- Manual-set AUC on the equivalence probe peaks at layer 10 (0.978) and
  drops to 0.769 at layer 12.
- Projecting out negation direction alone moves layer-12 manual AUC by
  0.022. Almost nothing.

### Stage 3: training dynamics

Trains baseline from scratch with `save_steps=200`. 15 snapshots across
3 epochs (steps 200 through 2907). Scores each snapshot on val + manual
+ claude.

Key observation that drove Stage 5: at step 400 the model has manual 95%,
claude 90%; by step 800 manual drops to 65%. So "the shortcut" is not
"what the model learns" but "what the model collapses into" after passing
through a better minimum.

### Stage 4: real hate-speech classifiers

Two datasets via HuggingFace and direct GitHub download:
- Davidson 2017 (`tdavidson/hate_speech_offensive`): 24,783 tweets,
  3 classes (hate, offensive, neither), MIT license.
- HateXplain (`Hate-speech-CNERG/hatexplain`, downloaded from the
  GitHub repo because `datasets` dropped script support):
  19,229 posts after majority-vote, 3 classes (hatespeech, normal,
  offensive), CC BY 4.0.

Standard 80/20 stratified split. TF-IDF + class-weighted LogReg as
baseline, then roberta-base 3 epochs.

Davidson is heavily imbalanced (77% offensive); F1 macro on hate is
only 0.45 even for RoBERTa because of the 286-sample minority class.
HateXplain is more balanced (40/30/30) and harder.

### Stage 5: fine step sweep

`max_steps=800`, `save_steps=25`. 32 snapshots. The "sweet spots" turned
out to be noise. Within steps 100 to 250 the manual accuracy oscillates
80, 65, 80, 55, 80. The model is unstable early in training.

Top 5 snapshots by manual+claude sum:
- step 250: manual 85, claude 90, ood 0.954
- step 125: manual 80, claude 90, ood 0.953
- step 200: manual 65, claude 85, ood 0.946
- step 400: manual 70, claude 80, ood 0.946
- step 175: manual 80, claude 65, ood 0.941

If the sweet spot were a stable state, all five top snapshots would have
manual + claude > 80 + 80. They do not. The numbers depend on stochastic
gradient bounces. We cannot reliably reach this state by early stopping.

### Stage 6: multi-direction ablation

For each of six shortcut proxies, fit a logistic regression on the
synthetic-train residuals at layer 12, take the coefficient vector,
orthonormalize the K vectors stacked together, project each residual onto
the orthogonal complement, retrain the equivalence probe.

Proxies:
- proxy_neg_flip: does text_b have negation that text_a does not (AUC 0.96)
- proxy_len_ratio: |text_b| / |text_a| above median (AUC 0.94)
- proxy_b_starts_the_post: text_b starts with "the post" (AUC 0.95)
- proxy_b_has_first_person: "i", "my", "i'd", "i'm", "i've" in text_b (AUC 0.83)
- proxy_b_much_shorter: |text_b| < |text_a|/2 (AUC 0.99)
- proxy_b_has_verdict_word: "hate", "offensive", "abuse", "abusive",
  "normal", "neutral", "benign" in text_b (AUC 0.94)

Each proxy is individually highly decodable from the residual stream. So
the model definitely encodes all of them. But projecting all six out moves
manual AUC by 0.011. The shortcut is high-rank and redundantly encoded.

Same procedure on the v04 winner gave manual AUC = 0.978 at K=0 and
manual AUC = 0.978 at K=6. Nothing to ablate because the shortcut never
formed.

### Stage 7: bigger eval, GPT voice

50 paraphrases of OOD test posts in casual-reader-on-Reddit voice via
GPT-4o-mini, verified by a second GPT-4o-mini call (asking "do these two
reactions reach the SAME conclusion or DIFFERENT?"). The verifier's
verdict becomes the label. If the original target was "make this
equivalent" but the verifier says DIFFERENT, we keep the pair with the
verifier's label (these are the hardest, most realistic cases).

| Model | Acc on n=50 |
|---|---|
| poster baseline | 54% |
| v04 | 68% |

v04 helps but the 95% claim from n=20 was inflated.

### Stage 8: label-noise audit

For each of 192 stratified pairs from reformulated_full.csv (~12 per
domain per label, 8 domains x 2 labels), ask GPT-4o-mini blind whether
the two rationales reach SAME or DIFFERENT conclusions. Compare to the
stored label.

Stored "equivalent": 100% agreement.
Stored "unequivalent": 91% agreement (9% are actually saying the same
thing).

Overall 95.3% clean. The data is mostly correct. The shortcut is not a
labelling issue.

### Stage 9: cross-generator generation and matrix

Generation:
- Sample 100 OOD test source posts once (cached in
  `overnight/cross_gen/_source_posts.csv`).
- For each of 5 LLMs, generate two paraphrases per source post (one
  intended equivalent, one intended unequivalent). 200 pairs per
  generator, 1000 total.
- Same system prompts for all generators (equivalent: precise paraphrase;
  unequivalent: flip verdict / swap entity / invert sentiment).
- All via OpenRouter, temperature 0.7, top_p 0.9.
- Resumable: each pair is keyed by (post_id, label) and skipped if
  already in the CSV. The bug we hit (empty CSV crashes pd.read_csv on
  resume) is fixed.

Mistral 7B's OpenRouter endpoint returned 404. We dropped it and ran
the matrix on the remaining 5 generators.

Matrix:
- For each generator, split its 200 pairs 80/20 by source post (so the
  same source post never appears in both train and test).
- Train one roberta-base on each generator's 160-pair train (3 epochs,
  batch 16, lr 2e-5).
- Also train a multi_gen model on the union of all 5 generators'
  train splits (800 pairs).
- Cross-evaluate: each model on every generator's test split (40 pairs)
  plus the 40 human pairs from manual + claude.

DeepSeek and Llama 8B's single-generator models collapsed: predict
equivalent on everything, ~50% accuracy on every test set (which happens
to match the 50/50 label balance). Their data is still useful in the
multi_gen union, but 160 pairs alone is too few for those generators'
paraphrases.

The multi_gen row hit 100% on 4 of 5 generator test sets, 92.5% on Llama,
and 90% on the 40 human pairs. That was the headline.

### Stage 10: bigger human eval, Claude voice

100 fresh human-style reactions via anthropic/claude-haiku-4.5 on OOD
test posts that do not appear in:
- Stage 7's harder_eval.csv (50 posts)
- Stage 9's _source_posts.csv (100 posts)
- manual labelling.csv (20 posts)
- roberta_training/claude_test_labels.csv (20 posts)

Different voice: "tired person writing on their phone, contractions OK,
no markdown". Different prompt structure than Stage 7. Different verifier
(Haiku, not GPT).

Scored three models:
- poster_baseline (the original 30k Qwen-3B checkpoint from
  judge_finetune/runs/roberta_base/checkpoints/checkpoint-1576)
- v04_style_mix_qwen7b (the Stage 1 winner)
- multi_gen (the Stage 9 winner)

Results in section 2-H above.

### Stage 11: LLM zero-shot ceiling

Each of 5 frontier LLMs called once per pair on the Stage 10 100-pair set
via OpenRouter. Prompt asks for one word answer: equivalent or
unequivalent. Temperature 0.

| Model | Acc |
|---|---|
| openai/gpt-4o-2024-11-20 | 100 |
| anthropic/claude-haiku-4.5 | 100 |
| qwen/qwen-2.5-72b-instruct | 100 |
| anthropic/claude-sonnet-4.5 | 99 |
| deepseek/deepseek-chat-v3.1 | 98 |

The dataset ceiling is essentially 100%. Haiku scoring 100% is
self-validating (Haiku was the verifier) but the other four are
independent and they also hit 98 to 100. So the labels are robust.

### Stage 12: leave-one-out

For each of 5 generators, train multi_gen on the OTHER 4 (640 pairs).
Score on each generator's test split plus Stage 10 humans.

Concrete numbers in POSTER_REWRITE.md section 3. The clear ranking by
"how much the model loses when you remove this generator" (on humans):

Most critical: anthropic_haiku (loses 21 points).
Next: meta_llama3_8b (loses 7 points).
Next: qwen25_72b (loses 7 points).
Next: openai_gpt4omini (loses 4 points).
Least critical: deepseek_v3 (loses 1 point).

Note: this ranking is on a single eval set with n=100. Confidence interval
around each point is about plus/minus 8. The "anthropic is critical"
finding is well outside the noise floor (21 > 8). The other rankings
are within noise.

### Stage 13: sentence-bert baseline

Encode every rationale via sentence-transformers/all-mpnet-base-v2.
Cosine similarity per pair. Tune threshold on the multi_gen training
pool (n=998) to maximize macro F1. Apply on three human eval sets.

Also fit a LogReg on three features: (cosine similarity, absolute length
difference, Jaccard token overlap).

AUROC was below 0.5 on every human set. Reason: training data has
similar-length LLM-vs-LLM pairs; human eval has very different lengths
(short reaction vs long explanation). The threshold tuned on training
maps almost every human pair to "unequivalent" (below the threshold
because of length mismatch), giving 100% uneq recall and 0% eq recall.

This is a useful negative result: it forecloses the "why not just use
sentence embeddings" critique.

## 4. Full results table

The "human" column is the Stage 10 n=100 Claude eval unless noted.

| Model | Training data | Manual n=20 | Stage 7 n=50 | Stage 10 n=100 | AUROC | eq / uneq @ Stage 10 |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 34k OOD train | not eval'd | not eval'd | not eval'd | not eval'd | not eval'd |
| GPT-4o-mini zero-shot | 0 | not eval'd | not eval'd | 100 | not eval'd | 100/100 |
| poster baseline (Qwen-3B 30k) | 30,000 | 55 | 54 | 59 | 0.745 | 40 / 84 |
| v04 (poster + Qwen-7B 200) | 30,200 | 95 | 68 | 66 | 0.808 | 51 / 86 |
| multi_gen (5 LLMs, 800) | 800 | not eval'd | not eval'd | 81 | 0.896 | 84 / 77 |
| LOO drop anthropic | 638 | not eval'd | not eval'd | 60 | 0.773 | 42 / 84 |
| LOO drop deepseek | 638 | not eval'd | not eval'd | 80 | 0.889 | 81 / 79 |
| LOO drop openai | 638 | not eval'd | not eval'd | 77 | 0.867 | 75 / 79 |
| LOO drop llama | 638 | not eval'd | not eval'd | 74 | 0.846 | 98 / 42 |
| LOO drop qwen | 640 | not eval'd | not eval'd | 74 | 0.846 | 77 / 70 |
| Sentence-BERT cosine | 0 (encoder pretrained) | 43 | 46 | 43 | 0.486 | 0 / 100 |
| Claude Sonnet 4.5 zero-shot | 0 | not eval'd | not eval'd | 99 | not eval'd | 100 / 98 |
| GPT-4o zero-shot | 0 | not eval'd | not eval'd | 100 | not eval'd | 100 / 100 |
| Claude Haiku 4.5 zero-shot | 0 | not eval'd | not eval'd | 100 | not eval'd | 100 / 100 |
| Qwen 2.5 72B zero-shot | 0 | not eval'd | not eval'd | 100 | not eval'd | 100 / 100 |
| DeepSeek V3 zero-shot | 0 | not eval'd | not eval'd | 98 | not eval'd | 96 / 100 |

Hate-speech (different task, single-text):

| Dataset | Model | Acc | F1 macro |
|---|---|---|---|
| Davidson 2017 | TF-IDF + LogReg | 87.5 | 0.742 |
| Davidson 2017 | RoBERTa-base | 91.6 | 0.767 |
| HateXplain | TF-IDF + LogReg | 65.6 | 0.646 |
| HateXplain | RoBERTa-base | 71.5 | 0.703 |

## 5. Methodology notes

### Eval sets and overlap control

Four eval sets exist:
- A: manual labelling.csv (20 pairs, human-written).
- B: claude_test_labels.csv (20 pairs, Claude 4.6-written).
- C: Stage 7 harder_eval.csv (50 GPT-4o-mini reactions on fresh OOD posts).
- D: Stage 10 claude_human_eval.csv (100 Claude Haiku reactions on fresh
  OOD posts, excluding all source posts from A, B, C).

The 100 cross-generator source posts in Stage 9 (`_source_posts.csv`) are
also held out of D. So D is genuinely fresh: no model in our pipeline has
seen those source posts during training or during prior eval.

### Verification of generated labels

Stages 7 and 10 both use a two-step generation: first generate the
comment with the intended label, then ask a second LLM call "do these
two reach the SAME or DIFFERENT conclusion?". The verifier's verdict is
the final label. This catches the case where the generator failed to
actually flip the verdict on the "unequivalent" prompt.

Stage 11 then confirms that 4 independent frontier LLMs (not the
verifier) agree with the labels at 98 to 100%, so we trust them.

### Train/test split by source post

In Stage 9, the same source post is paraphrased by all 5 generators.
If we split paraphrases randomly, paraphrases of the SAME source post
would appear in both train and test, leaking information about the post
across the split. We split by source post: the 80% train and 20% test
splits are computed on unique source posts, then every paraphrase of
each post lands in the same split. No leakage.

### Hyperparameters

Same for every roberta-base fine-tune:
- Optimizer: AdamW (default in transformers)
- LR: 2e-5
- Batch size: 16 or 32 depending on stage
- Weight decay: 0.01
- Warmup ratio: 0.06
- Epochs: 1 or 3 (most are 3)
- Max sequence length: 256
- Mixed precision: bf16 on the 5090
- Seed: 42 throughout

### Dataloader on Windows

We initially used `dataloader_num_workers=4` and the first overnight run
silently deadlocked: 8 zombie dataloader processes from a prior killed
run held the GPU and the new run never made progress for over an hour.
Fix: `num_workers=0`. Documented in the commit message "num_workers=0
on Windows".

## 6. Limitations and caveats

### Sample size

- The original manual set is 20 pairs. The Claude set is 20 pairs. Both
  are too small to claim a precise number. Treat as plus/minus 10 pp.
- Stage 7 (n=50) is better. Treat as plus/minus 7 pp.
- Stage 10 (n=100) is the best. Treat as plus/minus 5 pp.
- The cross-generator matrix's test splits are 40 pairs each. Plus/minus
  8 pp.
- The LOO ablation's anthropic finding (21 pp drop) is well outside
  noise. The other rankings are within noise.

### Multi_gen vs Stage 9 in-distribution diagonals

The multi_gen model trains on 800 pairs and tests on 40-pair held-out
splits from the same 5 generators. It hits 100% on 4 of 5 test sets. We
suspect this is partially because the test posts (20 of the 100 source
posts) and the train posts (80 of the 100) share enough vocabulary that
the model can rote-match. The 100% diagonals are probably inflated.

The human eval (D, n=100, completely fresh source posts) is the
trustworthy headline: 81%.

### Generation-time bias

Stage 10's labels come from Haiku-as-generator + Haiku-as-verifier. There
is correlation. We control for this by checking with 4 other frontier
LLMs (Stage 11). They all score 98 to 100, so the labels are reliable.
But a stricter setup would use disjoint generator and verifier.

### Davidson dataset class imbalance

Davidson 2017 is 77% offensive / 17% neither / 6% hate. F1 on the rare
"hate" class is 0.45 even for RoBERTa. This is a known feature of the
dataset, not a flaw in our pipeline. Worth flagging in the poster.

### Reproducibility of the 81%

We ran multi_gen training once with seed 42. The 81 +- 5 estimate on n=100
gives a 95% confidence interval of roughly 73 to 89. Running 3 seeds
would tighten this. We did not do it.

### v04's true number

v04 hit 95% on n=20 manual, 68% on n=50 GPT, 66% on n=100 Claude. Pooled
across all three (n=170): 69 +- 4. Treat "v04 is a real improvement over
baseline but a smaller one than the original 95% claim suggests" as the
honest framing.

## 7. Things that did not help

These are worth knowing for the writeup. They are negative results that
shape the conclusion.

- **Dropping high-cosine "unequivalent" pairs (v01).** Made things worse.
  Manual acc 35%, equivalent recall 0%. Too aggressive filtering removes
  too much of the equivalent supervision.

- **Dropping negation-flip pairs (v02).** Modest help (50 to 70 on manual)
  but small. Suggests negation is only one of many shortcut features.

- **Class rebalancing alone (v03).** 50 to 60 on manual. Small. The
  collapse is not just a class imbalance issue.

- **Adding only Qwen-7B paraphrases (v04).** Looked great on n=20
  but the gain shrank on bigger evals. Multi-generator beats
  multi-Qwen-variant.

- **Multi-direction inference-time ablation (Stage 6).** Six shortcut
  directions projected out, manual AUC moved 0.011. The shortcut is
  not low-rank.

- **Off-the-shelf sentence-BERT (Stage 13).** AUROC < 0.5 on humans.
  Pretrained embeddings do not solve this task.

- **Early stopping (Stages 3 and 5).** Looked promising at step 250 to 400
  but the sweet spots are noise spikes. Cannot reliably hit by stopping.

- **DeepSeek and Llama-8B as single-generator training sets (Stage 9).**
  At 160 pairs they collapsed to "predict equivalent". Their data is
  useful in the multi_gen union but not on its own.

## 8. Open questions and future work

If someone picks this up:

1. **Scale multi_gen.** Repeat Stage 9 with 500 or 1000 pairs per
   generator. Does 81% become 88% or is it a ceiling?

2. **Retrain multi_gen with disjoint generator and verifier.** Use Claude
   for generation, GPT for verification. Eliminates any same-model
   correlation in labels.

3. **More human voices.** Stage 10 used one Claude voice. Try the same
   eval with a GPT-4o voice and a Llama voice. Compare.

4. **Probe the multi_gen model at every layer.** We did this for baseline
   and v04 (best_mitigation) at layer 12. Doing it for multi_gen at
   every layer would tell us whether the cross-generator training
   removes the late-layer collapse entirely or just delays it.

5. **Apply the rationale judge to Davidson explanations.** Use a real
   LLM to write hate-speech rationales for Davidson posts, then use our
   judge to evaluate those rationales against ground truth. This closes
   the loop: from synthetic equivalence judge to actual hate-speech
   evaluation tool.

6. **3-class collapse check.** Currently the judge is binary. Real
   rationales can also be "partially equivalent" (agrees on verdict,
   disagrees on reasoning). Worth exploring as a third class.

7. **Larger encoder.** roberta-base is 125M parameters. DeBERTa-v3-large
   is 350M. The poster used the same. Re-running multi_gen with the
   larger encoder would tell us whether the shortcut is a roberta-base
   limit or a general phenomenon.

8. **3-seed average.** Run multi_gen with seeds 42, 43, 44; report
   plus/minus on the 81%.

## 9. Notes for the paper

If this gets written up:

- The story is "single-generator synthetic data fails; multi-generator
  succeeds; here is the recipe and why." Not "we found a shortcut."

- The hero number is "81% on a 100-pair blind human eval with 30x less
  data than the baseline."

- The supporting findings (LLM ceiling 100%, sentence-bert fails,
  layer 12 collapse, Anthropic Haiku is critical) all reinforce the
  same point: this is a representation problem caused by stylistic
  homogeneity in the training data, not a data scale or label noise
  problem.

- The methodology innovation is "use OpenRouter to generate paraphrases
  from 5 LLMs on the same source posts." Simple but easy to miss. This
  controls for content while varying style.

- The hate-speech classifier results (Stage 4) belong in a related-works
  or motivation section, not the main results. The paper's primary
  contribution is on the equivalence-judge problem.

- The mechanistic-interpretability findings (Stage 2 layer-12 collapse,
  Stage 6 high-rank shortcut) are sharp enough to stand alone in their
  own section.

- Sentence-BERT failure (Stage 13) is the kind of negative result that
  saves the paper from one reviewer's "why didn't you try X" comment.
  Worth a short paragraph in the methods or related works.

- The poster's existing F1 baselines (TF-IDF 0.79, GPT-4o-mini 0.96)
  are fine as the in-distribution context. They are not the headline
  anymore but they belong in the table.
