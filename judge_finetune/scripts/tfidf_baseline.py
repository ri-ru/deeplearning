import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DataConfig, load_and_split

train_ds, val_ds, test_ds = load_and_split(DataConfig(csv_path="../reformulated_full.csv"))
train_df = train_ds.to_pandas()
test_df  = test_ds.to_pandas()

train_text = train_df["text_a"] + " [SEP] " + train_df["text_b"]
test_text  = test_df["text_a"]  + " [SEP] " + test_df["text_b"]

vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train = vec.fit_transform(train_text)
X_test  = vec.transform(test_text)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_df["label"])
preds = clf.predict(X_test)

print("=== TF-IDF + LogReg (no neural network, no pre-training) ===")
print("Overall  acc={:.3f}  f1={:.3f}".format(
    accuracy_score(test_df["label"], preds),
    f1_score(test_df["label"], preds, average="macro")))
for dom in ["anatomy", "gab", "toxigen"]:
    mask = test_df["domain"] == dom
    print("{}  acc={:.3f}  f1={:.3f}".format(
        dom,
        accuracy_score(test_df["label"][mask], preds[mask]),
        f1_score(test_df["label"][mask], preds[mask], average="macro")))
