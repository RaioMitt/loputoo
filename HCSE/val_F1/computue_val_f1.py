import time
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ==========================================
# 1. SETTINGS AND VOTING PARAMETERS
# ==========================================
HF_API_KEY = "MY_HF_KEY"
HF_BASE_URL = "https://router.huggingface.co/v1"

# Select the model for which we compute the val_F1 weight
MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"

# STEP 1: Query N times
N_ITERATIONS = 5  # How many times to query the model with the same prompt


def setup_llm_client():
    return ChatOpenAI(
        openai_api_key=HF_API_KEY,
        openai_api_base=HF_BASE_URL,
        model_name=MODEL_NAME,
        temperature=0.5,
        max_tokens=500,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )


def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# STEP 2: Intra-model voting (Majority Vote)
def get_majority_vote(predictions):
    """
    Takes a list of predictions, e.g.: [('Jailbreak', 'Toxic'), ('Jailbreak', 'Toxic'), ('Safe', 'Safe_Community')]
    and returns the most frequently occurring pair.
    """
    if not predictions:
        return "ERROR", "ERROR"

    # Count how many times each L1, L2 pair occurred
    counts = Counter(predictions)
    # Return the most common pair
    most_common_pair = counts.most_common(1)[0][0]
    return most_common_pair


def run_validation_voting(dataset_path, output_path, chain):
    try:
        df = pd.read_parquet(dataset_path)
        print(f"Loaded {len(df)} rows from {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    true_l1, true_l2 = [], []
    final_pred_l1, final_pred_l2 = [], []

    for index, row in df.iterrows():
        user_input = str(row['prompt'])

        real_l1 = str(row['L1_label']).strip().capitalize()
        real_l2 = str(row['community']).strip()
        if real_l1 == 'Safe':
            real_l2 = 'Safe_Community'

        true_l1.append(real_l1)
        true_l2.append(real_l2)

        print(f"\nRow {index + 1}/{len(df)} [Actual: {real_l1} | {real_l2}]")

        current_row_predictions = []

        # Query the model N times
        for i in range(N_ITERATIONS):
            try:
                output = chain.invoke({'question': user_input})

                p_l1, p_l2 = "ERROR", "ERROR"
                for line in output.split('\n'):
                    line = line.strip()
                    if line.startswith('L1:'):
                        p_l1 = line.replace('L1:', '').strip().capitalize()
                    elif line.startswith('L2:'):
                        p_l2 = line.replace('L2:', '').strip()

                if p_l1 == "ERROR" or p_l2 == "ERROR":
                    current_row_predictions.append(("ERROR", "ERROR"))
                else:
                    current_row_predictions.append((p_l1, p_l2))

                print(f"  Vote {i + 1}/{N_ITERATIONS}: {p_l1} | {p_l2}")

            except Exception as e:
                print(f"  Vote {i + 1} ERROR (API): {e}")
                current_row_predictions.append(("ERROR", "ERROR"))
                time.sleep(1)

        # Apply majority vote across N responses (Step 2)
        voted_l1, voted_l2 = get_majority_vote(current_row_predictions)
        print(f"--> FINAL DECISION (Majority Vote): {voted_l1} | {voted_l2}")

        final_pred_l1.append(voted_l1)
        final_pred_l2.append(voted_l2)

    # STEP 4: Compare with ground truth
    valid_indices = [i for i, p in enumerate(final_pred_l1) if p in ['Jailbreak', 'Safe']]

    print("\n" + "=" * 60)
    print(f"VALIDATION WEIGHT (val_F1) RESULTS: {MODEL_NAME}")
    print(f"Sampling: {N_ITERATIONS} times per prompt (Majority Voting)")
    print("=" * 60)

    if len(valid_indices) > 0:
        y_true_l2 = [true_l2[i] for i in valid_indices]
        y_pred_l2 = [final_pred_l2[i] for i in valid_indices]

        # STEPS 5 & 6: Compute F1 per class and Macro-F1 (val_F1)
        # zero_division=0 ensures that if the model misses a class, it scores 0.
        val_f1 = f1_score(y_true_l2, y_pred_l2, average='macro', zero_division=0)

        print(f"\n✅ FINAL MODEL WEIGHT (val_F1 at L2 level): {val_f1:.4f}\n")

        print("--- Detailed class breakdown (Precision, Recall, F1) ---")
        print(classification_report(y_true_l2, y_pred_l2, zero_division=0))
    else:
        print("ERROR: No valid predictions could be voted on.")

    # Save results for later analysis
    df['Voted_L1'] = final_pred_l1
    df['Voted_L2'] = final_pred_l2
    df.to_parquet(output_path)
    print(f"\nValidation voting results saved to: {output_path}")


def main():
    llm = setup_llm_client()
    template_content = load_template("../../template2.pmt")
    prompt = PromptTemplate(input_variables=['question'], template=template_content)
    chain = prompt | llm | StrOutputParser()

    # USE THE VALIDATION SET! (Test data must not be touched here)
    input_dataset = "val_dataset_split.parquet"
    output_dataset = f"validation_voting_{MODEL_NAME.split('/')[-1]}.parquet"

    run_validation_voting(input_dataset, output_dataset, chain)


if __name__ == "__main__":
    main()