import pandas as pd
df = pd.read_csv(r'overnight/cross_gen/anthropic_haiku.csv')
print(f'rows: {len(df)}  cols: {list(df.columns)}')
print(f'label balance: {dict(df["meaning"].value_counts())}')
print()
print('--- equivalent example ---')
r = df[df['meaning']=='equivalent'].iloc[0]
print('explanation:', r['explanation'][:240])
print()
print('paraphrase :', r['synthetic'][:240])
print()
print('--- unequivalent example ---')
r = df[df['meaning']=='unequivalent'].iloc[0]
print('explanation:', r['explanation'][:240])
print()
print('paraphrase :', r['synthetic'][:240])
