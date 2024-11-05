import numpy as np
import pandas as pd

df = {
    "light_years" : [23, 54, 24, 87, 50, 1, 100],
    "visible" : [1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(df)

df['X_2'] = df['light_years'] ** 2
df['Y_2'] = df["visible"] ** 2
df['XY'] = df['light_years'] * df["visible"]
n = len(df)
a = ((sum(df["visible"]) * sum(df['X_2'])) - ((sum(df['light_years'])) * sum(df['XY']))) / ((n * sum(df['X_2'])) - (sum(df['light_years']) ** 2))
b = ((n * sum(df['XY'])) - (sum(df['light_years']) * sum(df["visible"]))) / ((n * sum(df['X_2'])) - (sum(df['light_years']) **2))

e = 2.71828
den = (1 + e ** (-(b * 60 + a)))
vis_prob = 1 / den
print("Probability of visibility is:", round(vis_prob, 4))
