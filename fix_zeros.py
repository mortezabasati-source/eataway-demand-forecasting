import pandas as pd

df = pd.read_csv("trainable_data.csv")
test_wks = sorted(df["year_week"].unique())[-12:-6]  # Test set W30-W35 roughly
test_df = df[df["year_week"].isin(test_wks)].copy()

actual_zero = (test_df["faktisk"] == 0).sum()
print(f"Total actual zeros in test set: {actual_zero}")

# Let's see how many of these ACTUAL ZEROS had strong lag features
strong_history = test_df[(test_df["faktisk"] == 0) & (test_df["rolling_mean_4w"] >= 2.0)]
print(f"Zeros with rolling_mean_4w >= 2: {len(strong_history)} cases")
print(f"Percentage of uncatchable zeros: {len(strong_history)/actual_zero:.1%}")

# And let's see how many zeros W36 actually has, given W37 is acting up
if "2026-W36" in test_df["year_week"].values:
    w36_zeros = (test_df[test_df["year_week"] == "2026-W36"]["faktisk"] == 0).sum()
    print(f"Zeros in W36: {w36_zeros}")
