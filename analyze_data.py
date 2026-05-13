import pandas as pd
df = pd.read_csv('C:/Users/jan/downloads/deep learning/reformulated_full.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print()
print('meaning counts:')
print(df['meaning'].value_counts())
print()
print('domain counts:')
print(df['domain'].value_counts())
print()

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
all_text = pd.concat([df['explanation'], df['synthetic']]).fillna('')
vec.fit(all_text)

eq = df[df['meaning'] == 'equivalent'].sample(500, random_state=42)
uneq = df[df['meaning'] == 'unequivalent'].sample(500, random_state=42)

def avg_cosine(subset):
    a = vec.transform(subset['explanation'].fillna(''))
    b = vec.transform(subset['synthetic'].fillna(''))
    sims = [cosine_similarity(a[i], b[i])[0][0] for i in range(a.shape[0])]
    return sum(sims)/len(sims)

print('Avg TF-IDF cosine sim (equivalent):', round(avg_cosine(eq), 3))
print('Avg TF-IDF cosine sim (unequivalent):', round(avg_cosine(uneq), 3))
