import pandas as pd
import requests
import time

# File names
file_adversarial = "../eng/jailbreak.parquet"
file_safe = "../eng/regular.parquet"

print(f"Loading data from {file_adversarial} and {file_safe}...")

# 1. ADVERSARIAL INPUT SAMPLE (55 rows from jailbreak.parquet)
df_adversarial = pd.read_parquet(file_adversarial)
# Filter out rows where the prompt is 7000 characters or longer
df_adversarial = df_adversarial[df_adversarial['prompt'].str.len() < 7000]
# Drop rows with missing community labels
df_adversarial = df_adversarial.dropna(subset=['community'])
# Shuffle and take 5 rows from each community (11 * 5 = 55)
sample_adversarial = df_adversarial.sample(frac=1, random_state=42).groupby('community', group_keys=False).head(5).reset_index(drop=True)
sample_adversarial['jailbreak'] = True

# 2. SAFE INPUT SAMPLE (45 rows from regular.parquet)
df_safe = pd.read_parquet(file_safe)
# Filter out prompts that are too long here as well
df_safe = df_safe[df_safe['prompt'].str.len() < 7000]
# Take 45 random rows
sample_safe = df_safe.sample(n=45, random_state=42).reset_index(drop=True)
sample_safe['community'] = "Regular/Safe"
sample_safe['jailbreak'] = False

# 3. MERGE DATASETS (100 rows total)
combined_df = pd.concat([sample_adversarial, sample_safe], ignore_index=True)
# Shuffle row order
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Datasets merged and overly long texts removed. Total rows for translation: {len(combined_df)}")


# 4. TARTUNLP TRANSLATION FUNCTION
def translate_to_estonian(text):
    url = "https://api.tartunlp.ai/translation/v2"
    # Safety measure: in case anything longer slips through
    cleaned_text = str(text)

    payload = {
        "text": cleaned_text,
        "src": "eng",
        "tgt": "est",
        "domain": "general",
        "application": "bakatoo_andmestik"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get('result', text)
        else:
            print(f"\nAPI error: {response.status_code}")
            return text
    except Exception as e:
        print(f"Request failed: {e}")
        return text


# 5. TRANSLATION LOOP
translated_texts = []

print("Starting machine translation (this may take approx. 1-2 minutes)...")
for index, row in combined_df.iterrows():
    print(f"Translating {index + 1}/{len(combined_df)} | Category: {row['community']}")
    translation = translate_to_estonian(row['prompt'])
    translated_texts.append(translation)
    time.sleep(0.5)

# 6. FORMAT FINAL TABLE
final_dataset = pd.DataFrame({
    'ID': range(1, len(combined_df) + 1),
    'Community/Category': combined_df['community'],
    'Eng': combined_df['prompt'],
    'Est': translated_texts,
    'Corrected Est': "",
    'Note (What I changed)': "",
    'Jailbreak': combined_df['jailbreak']
})

# 7. SAVE RESULTS
output_filename = "dataset_for_manual_review_100.csv"
final_dataset.to_csv(output_filename, index=False, encoding='utf-8')

print(f"\nAll done! Your 100-row working file is saved here: '{output_filename}'")