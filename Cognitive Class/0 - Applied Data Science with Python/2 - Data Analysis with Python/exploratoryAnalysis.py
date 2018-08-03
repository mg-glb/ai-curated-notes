import numpy as np
import pandas as pd

filename ="clean_df.csv"
df = pd.read_csv(filename)
df_test = df.filter(items=["body-style","price"])
df_grp = df_test.groupby(["body-style"], as_index=False).mean()

print(df_grp['price'])