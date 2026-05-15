# Test sets

Two human-curated eval sets, 20 rows each.

## `manual_labelling.csv` (lives in `../synthetic_data_generation_improved/`)

20 posts I hand-labelled. Columns: post, the AI explanation, my own first-
person comment, and equivalent/unequivalent. 13 equivalent, 7 unequivalent.

Important: this file is ALSO the few-shot seed inside `script.py` -
`load_fewshot_examples` reads it and injects rows verbatim into the Qwen
generation prompt. So evaluating the trained RoBERTa on this same file is
not a clean held-out test. The training distribution was conditioned on
these 20 rows. For a real held-out result, keep ~10 of them out of the
few-shot pool.

## `claude_test_labels.csv`

22 rows generated with Claude 4.6 using the same few-shot prompt template
from `synthetic_data_generation_improved/script.py`, then human-checked.
Independent of `manual_labelling.csv`.

Observed result (eval script not in git yet): RoBERTa got 10/10 on
unequivalent rows but only 3/10 on equivalent rows. The Claude voice
("This post is...") is stylistically different from Qwen's voice ("I find
it...") and the trained model treats anything in the Claude register as
unequivalent by default.

## Gaps

- Which posts ended up in each set, how disagreements got resolved, which
  Claude model exactly was used - only Flavia knows. Will need to write
  this in.
