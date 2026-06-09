"""How well does a length-only, keyword-only, or random-LogReg classifier
do on the same 20 pairs?  This sanity-checks multi_gen's 95%."""
import sys, re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from overnight.eval_common import LABEL2ID, load_human_csv

rows = load_human_csv(REPO / "manual labelling.csv")
texts_a = [r["explanation"] for r in rows]
texts_b = [r["comment"] for r in rows]
labels = np.array([LABEL2ID[r["label"]] for r in rows])
n = len(rows)
print(f"n={n}, balance: eq={(labels==0).sum()}, uneq={(labels==1).sum()}")
print(f"random-guess accuracy if always predict eq: {(labels==0).mean()*100:.0f}%")
print()

# 1) length-only
len_b = np.array([len(b) for b in texts_b])
print(f"len(text_b) eq mean = {len_b[labels==0].mean():.0f},  uneq mean = {len_b[labels==1].mean():.0f}")
# best length threshold
best_t, best_acc = 0, 0
for t in range(20, 400, 5):
    preds = (len_b > t).astype(int)
    a = (preds == labels).mean()
    if a > best_acc:
        best_acc = a; best_t = t
print(f"best-length-threshold accuracy: {best_acc*100:.0f}% at t={best_t} chars")

# 2) keyword: count words that look like disagreement
DISAGREE_RE = re.compile(r"\b(hateful|hate speak|transphobia|disrespectful|not ok|"
                          r"i don'?t appreciate|not normal|harmful|offensive|"
                          r"i think|i would say|i wouldn'?t say|i feel)\b", re.I)
counts = np.array([len(DISAGREE_RE.findall(b)) for b in texts_b])
preds = (counts >= 1).astype(int)
print(f"keyword-rule accuracy: {(preds == labels).mean()*100:.0f}%  "
      f"(predict uneq if any disagreement keyword)")

# 3) TF-IDF on comment alone, leave-one-out
print()
print("LOO TF-IDF + LogReg on comment text alone:")
correct = 0
for i in range(n):
    keep = [j for j in range(n) if j != i]
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_tr = vec.fit_transform([texts_b[j] for j in keep])
    X_te = vec.transform([texts_b[i]])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, labels[keep])
    correct += int(clf.predict(X_te)[0] == labels[i])
print(f"  accuracy: {correct/n*100:.0f}%  ({correct}/{n})")

# 4) Bag-of-words on (explanation + comment) LOO
print()
print("LOO TF-IDF + LogReg on (explanation + comment):")
joined = [a + " [SEP] " + b for a, b in zip(texts_a, texts_b)]
correct = 0
for i in range(n):
    keep = [j for j in range(n) if j != i]
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_tr = vec.fit_transform([joined[j] for j in keep])
    X_te = vec.transform([joined[i]])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, labels[keep])
    correct += int(clf.predict(X_te)[0] == labels[i])
print(f"  accuracy: {correct/n*100:.0f}%  ({correct}/{n})")
