# Overnight training log

Append-only log of every interesting training run. Each entry: timestamp,
variant name, key numbers, one-line interpretation.

Format:
```
## YYYY-MM-DD HH:MM  <variant>
- recipe: ...
- OOD F1: ...  manual_acc: ...  claude_acc: ...
- interpretation: ...
```

---

## 2026-05-17 22:25  v05_one_epoch  (smoke test)
- recipe: baseline data, only 1 epoch (instead of 3)
- OOD F1: 0.968  manual_acc: 50.0% (eq=23%/uneq=100%)  claude_acc: 65% (85% predicted unequivalent)
- interpretation: **the shortcut is already fully present after just 1 epoch.**
  OOD F1 matches the 3-epoch poster number (0.968) and the manual-set bias
  (predict 'unequivalent' on anything non-Qwen-style) is just as severe.
  Training longer doesn't *create* the shortcut — it cements it.
- file: `overnight/results/stage1/v05_one_epoch.json`

