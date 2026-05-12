import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA
# ==========================================
df_jailbreak = pd.read_parquet("jailbreak.parquet")
df_safe = pd.read_parquet("regular.parquet")

# Clean jailbreak data
df_jailbreak = df_jailbreak.dropna(subset=['community']).copy()
jailbreak_count = len(df_jailbreak)  # Expected: 406

# Sample the same number of rows from the safe dataset
df_safe_sampled = df_safe.sample(n=jailbreak_count, random_state=42).copy()
df_safe_sampled['community'] = 'Safe_Community'

# Add L1 labels
df_jailbreak['L1_label'] = 'Jailbreak'
df_safe_sampled['L1_label'] = 'Safe'

# ==========================================
# 2. BUILD TRAINING SET (2 rows per L2 class)
# ==========================================
# Group by community and take exactly 2 rows from each.
# (Using min(2, len(x)) in case some classes have only 1 row)
jb_train = df_jailbreak.groupby('community', group_keys=False).apply(
    lambda x: x.sample(n=min(2, len(x)), random_state=42)
)

# Remove these 22 rows from the remaining jailbreak data to prevent data leakage
jb_remaining = df_jailbreak.drop(jb_train.index)

# Sample the same number of rows (22) from the safe dataset
safe_train = df_safe_sampled.sample(n=len(jb_train), random_state=42)
safe_remaining = df_safe_sampled.drop(safe_train.index)

# ==========================================
# 3. SPLIT TEST AND VALIDATION SETS
# ==========================================
# 384 rows remain from each class.
# 321 go to test, the remaining 63 go to validation.

# Split jailbreak data (attempt stratification by L2 class)
try:
    jb_val, jb_test = train_test_split(
        jb_remaining,
        test_size=321,
        stratify=jb_remaining['community'],
        random_state=42
    )
except ValueError:
    print("Warning: Stratified split of remaining jailbreak data failed (some classes too small). Splitting randomly.")
    jb_val, jb_test = train_test_split(jb_remaining, test_size=321, random_state=42)

# Split safe data
safe_val, safe_test = train_test_split(safe_remaining, test_size=321, random_state=42)

# ==========================================
# 4. MERGE AND SHUFFLE DATASETS
# ==========================================
# Concatenate and shuffle (sample(frac=1) randomizes row order)
df_train = pd.concat([safe_train, jb_train], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df_val = pd.concat([safe_val, jb_val], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df_test = pd.concat([safe_test, jb_test], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# 5. VERIFY AND SAVE
# ==========================================
print("=== DATASET SPLIT SUMMARY ===")
print(f"Total dataset: {len(df_train) + len(df_val) + len(df_test)} rows (expected: 812)")
print("-" * 45)

print(f"TRAIN set: {len(df_train)} rows")
print(df_train["L1_label"].value_counts())
print("\nDetailed jailbreak breakdown in training set (should be 2 per class):")
print(df_train[df_train["L1_label"] == "Jailbreak"]["community"].value_counts())
print("-" * 45)

print(f"VALIDATION set: {len(df_val)} rows")
print(df_val["L1_label"].value_counts())
print("-" * 45)

print(f"TEST set: {len(df_test)} rows")
print(df_test["L1_label"].value_counts())
print("-" * 45)

# Save
df_train.to_parquet("train_dataset_split.parquet")
df_val.to_parquet("val_dataset_split.parquet")
df_test.to_parquet("test_dataset_split.parquet")
print("\n✅ Split files saved successfully!")