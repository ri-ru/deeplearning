import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('/mnt/c/Users/jan/downloads/deep learning/reformulated_full.csv')

print(f"Total rows: {len(df)}")
print(f"\nClass balance:")
print(df['meaning'].value_counts())
print(f"\nDomain counts:")
print(df['domain'].value_counts())

# Fit TF-IDF on all text
vec = TfidfVectorizer()
vec.fit(pd.concat([df['explanation'], df['synthetic']]).fillna(''))

# Compute pairwise cosine sim between explanation and synthetic for each row
a = vec.transform(df['explanation'].fillna(''))
b = vec.transform(df['synthetic'].fillna(''))

sims = []
for i in range(a.shape[0]):
    sims.append(cosine_similarity(a[i], b[i])[0][0])
df['cosine_sim'] = sims

eq   = df[df['meaning'] == 'equivalent']['cosine_sim']
uneq = df[df['meaning'] == 'unequivalent']['cosine_sim']

print(f"\n--- Cosine similarity: equivalent pairs (n={len(eq)}) ---")
print(f"  mean:   {eq.mean():.3f}")
print(f"  median: {eq.median():.3f}")
print(f"  std:    {eq.std():.3f}")
print(f"  <0.5:   {(eq < 0.5).mean()*100:.1f}%")
print(f"  >0.9:   {(eq > 0.9).mean()*100:.1f}%")

print(f"\n--- Cosine similarity: unequivalent pairs (n={len(uneq)}) ---")
print(f"  mean:   {uneq.mean():.3f}")
print(f"  median: {uneq.median():.3f}")
print(f"  std:    {uneq.std():.3f}")
print(f"  <0.5:   {(uneq < 0.5).mean()*100:.1f}%")
print(f"  >0.9:   {(uneq > 0.9).mean()*100:.1f}%")

# Histogram buckets
print(f"\n--- Distribution (equivalent) ---")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels = ['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0']
eq_hist   = pd.cut(eq,   bins=bins, labels=labels).value_counts().sort_index()
uneq_hist = pd.cut(uneq, bins=bins, labels=labels).value_counts().sort_index()
print(eq_hist.to_string())
print(f"\n--- Distribution (unequivalent) ---")
print(uneq_hist.to_string())
