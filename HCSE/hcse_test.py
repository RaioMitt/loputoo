import time
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ==========================================
# 1. SETTINGS AND FIXED VALIDATION SCORES
# ==========================================
HF_API_KEY = "MY_HF_KEY"
HF_BASE_URL = "https://router.huggingface.co/v1"

MODELS = {
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
    "DeepSeek": "deepseek-ai/DeepSeek-R1",
    "Llama3": "meta-llama/Meta-Llama-3-70B-Instruct"
}

# Number of times to query the model on test data (to find confidence)
N_RUNS = 5

# Fixed validation phase F1-scores (val_F1) for each model
VAL_F1_SCORES = {
    "Qwen": 0.2225,
    "DeepSeek": 0.2165,
    "Llama3": 0.1566
}


# ==========================================
# 2. INITIALIZE MODELS
# ==========================================
def setup_clients():
    return {
        "Qwen": ChatOpenAI(openai_api_key=HF_API_KEY, openai_api_base=HF_BASE_URL, model_name=MODELS["Qwen"],
                           temperature=0.8, top_p=0.95, max_tokens=500),
        "DeepSeek": ChatOpenAI(openai_api_key=HF_API_KEY, openai_api_base=HF_BASE_URL, model_name=MODELS["DeepSeek"],
                               temperature=0.5, top_p=0.9, max_tokens=500),
        "Llama3": ChatOpenAI(openai_api_key=HF_API_KEY, openai_api_base=HF_BASE_URL, model_name=MODELS["Llama3"],
                             temperature=0.7, top_p=0.9, max_tokens=500)
    }


def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# ==========================================
# 3. HCSE STAGES 2 & 3: VOTING LOGIC
# ==========================================
def perform_stage2_intra_model_voting(model_raw_runs):
    """
    Stage 2: Finds the model's majority vote over N runs,
    calculates confidence (k/N), and finds dynamic weight (val_F1 * conf).
    """
    model_results = {}
    for model, runs in model_raw_runs.items():
        counts = Counter(runs)
        majority_tuple, count = counts.most_common(1)[0]

        confidence = count / N_RUNS
        weight = VAL_F1_SCORES[model] * confidence  # Uses global fixed val_F1 score

        model_results[model] = {
            "prediction": majority_tuple,
            "confidence": confidence,
            "weight": weight
        }
    return model_results


def perform_stage3_hcse_voting(model_results):
    """
    Stage 3: Hierarchical weighted voting (L1 -> L2).
    """
    # STEP A: Decide L1 level (Sum the weights)
    l1_scores = {}
    for model, data in model_results.items():
        l1, l2 = data["prediction"]
        if l1 != "ERROR":
            l1_scores[l1] = l1_scores.get(l1, 0.0) + data["weight"]

    if not l1_scores:
        return "ERROR", "ERROR"

    winning_l1 = max(l1_scores, key=l1_scores.get)

    # STEP B: Decide L2 level (Conditional voting)
    l2_scores = {}
    for model, data in model_results.items():
        l1, l2 = data["prediction"]
        # ONLY models that voted for the winning L1 category can vote for L2
        if l1 == winning_l1 and l2 != "ERROR":
            l2_scores[l2] = l2_scores.get(l2, 0.0) + data["weight"]

    if winning_l1 == 'Safe':
        winning_l2 = 'Safe_Community'
    else:
        winning_l2 = max(l2_scores, key=l2_scores.get) if l2_scores else "ERROR"

    return winning_l1, winning_l2


# ==========================================
# 4. MAIN PROCESS: ZERO-SHOT TESTING
# ==========================================
def run_hcse_test(dataset_path, output_path, clients, base_template_text):
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} rows from the test dataset ({dataset_path})")

    # ZERO-SHOT Prompt Template
    prompt = PromptTemplate(input_variables=['question'], template=base_template_text)

    chains = {name: prompt | client | StrOutputParser() for name, client in clients.items()}

    true_l1, true_l2 = [], []
    final_pred_l1, final_pred_l2 = [], []
    all_raw_predictions = []

    for index, row in df.iterrows():
        user_input = str(row['prompt'])
        real_l1 = str(row['L1_label']).strip().capitalize()
        real_l2 = 'Safe_Community' if real_l1 == 'Safe' else str(row['community']).strip()

        true_l1.append(real_l1)
        true_l2.append(real_l2)

        print(f"\n--- ROW {index + 1}/{len(df)} [Actual: {real_l1} | {real_l2}] ---")

        # Stage 1: Find model confidence (Self-Consistency Sampling)
        model_raw_runs = {name: [] for name in clients.keys()}

        for model_name, chain in chains.items():
            print(f"  Querying model {model_name} {N_RUNS} times...")
            for i in range(N_RUNS):
                try:
                    output = chain.invoke({'question': user_input})
                    p_l1, p_l2 = "ERROR", "ERROR"

                    for line in output.split('\n'):
                        line = line.strip()
                        # Eemaldasin JSON-i jaoks vajadusel jutumärgid ja komad
                        if '"L1":' in line or 'L1:' in line:
                            p_l1 = line.replace('"L1":', '').replace('L1:', '').replace('"', '').replace(',',
                                                                                                         '').strip().capitalize()
                        elif '"L2":' in line or 'L2:' in line:
                            p_l2 = line.replace('"L2":', '').replace('L2:', '').replace('"', '').replace(',',
                                                                                                         '').strip()

                    if p_l1 == 'Safe':
                        p_l2 = 'Safe_Community'

                    # --- UUS DEBUG LOOGIKA ---
                    if p_l1 == "ERROR" or p_l2 == "ERROR":
                        print(f"    [Katse {i + 1}] Vormingu viga! Mudel vastas nii:")
                        print(f"      {output}")
                    # -------------------------

                    model_raw_runs[model_name].append((p_l1, p_l2))

                except Exception as e:
                    print(f"    [Katse {i + 1}] SÜSTEEMNE VIGA: {e}")
                    model_raw_runs[model_name].append(("ERROR", "ERROR"))

                time.sleep(1)  # Respecting API limits

        # Stage 2: Intra-model voting and weight calculation
        model_results = perform_stage2_intra_model_voting(model_raw_runs)

        for model_name, data in model_results.items():
            pred = data['prediction']
            conf = data['confidence']
            w = data['weight']
            print(f"  [{model_name}] Majority: {pred[0]} | {pred[1]} (Conf: {conf:.2f}, Weight: {w:.4f})")

        all_raw_predictions.append(str(model_results))

        # Stage 3: Mathematical HCSE hierarchical voting
        voted_l1, voted_l2 = perform_stage3_hcse_voting(model_results)
        print(f"==> HCSE FINAL DECISION: {voted_l1} | {voted_l2}")

        final_pred_l1.append(voted_l1)
        final_pred_l2.append(voted_l2)

    # ==========================================
    # 5. CALCULATE FINAL RESULTS
    # ==========================================
    valid_indices = [i for i, p in enumerate(final_pred_l1) if p in ['Jailbreak', 'Safe']]

    print("\n" + "=" * 60)
    print("HCSE ENSEMBLE FINAL TEST RESULTS")
    print("=" * 60)

    if len(valid_indices) > 0:
        y_true_l1 = [true_l1[i] for i in valid_indices]
        y_pred_l1 = [final_pred_l1[i] for i in valid_indices]
        y_true_l2 = [true_l2[i] for i in valid_indices]
        y_pred_l2 = [final_pred_l2[i] for i in valid_indices]

        print("\n--- L1 (Jailbreak vs Safe) ---")
        print(f"Accuracy: {accuracy_score(y_true_l1, y_pred_l1):.4f}")
        print(f"Macro-F1: {f1_score(y_true_l1, y_pred_l1, average='macro', zero_division=0):.4f}")

        print("\n--- L2 (Attack Communities) ---")
        print(f"Accuracy: {accuracy_score(y_true_l2, y_pred_l2):.4f}")
        print(f"Macro-F1: {f1_score(y_true_l2, y_pred_l2, average='macro', zero_division=0):.4f}")

        print("\n--- L2 Detailed Report ---")
        print(classification_report(y_true_l2, y_pred_l2, zero_division=0))
    else:
        print("ERROR: Could not make any valid predictions.")

    # Save results to database
    df['HCSE_L1'] = final_pred_l1
    df['HCSE_L2'] = final_pred_l2
    df['HCSE_Dynamic_Data'] = all_raw_predictions
    df.to_parquet(output_path)
    print(f"\n✅ HCSE final table saved: {output_path}")


def main():
    clients = setup_clients()
    base_template = load_template("template2.pmt")

    input_dataset = "test_dataset_split.parquet"
    output_dataset = "hcse_final_results_zero_shot.parquet"

    run_hcse_test(input_dataset, output_dataset, clients, base_template)


if __name__ == "__main__":
    main()