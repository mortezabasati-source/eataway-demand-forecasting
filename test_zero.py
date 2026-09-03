import pandas as pd
import numpy as np
df = pd.read_csv("trainable_data.csv")
print("Target total zero ratio:", (df["faktisk"] == 0).mean())
zero_cases = df[df["faktisk"] == 0]
print("Avg rolling_mean_4w when faktisk=0:", zero_cases["rolling_mean_4w"].mean())
print("Ratio where rolling_mean_4w > 1 but actually 0:", (zero_cases["rolling_mean_4w"] > 1).mean())
