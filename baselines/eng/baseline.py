import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 1. SETTINGS
HF_API_KEY = "MY_HF_KEY"
HF_BASE_URL = "https://router.huggingface.co/v1"

MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"


def setup_llm_client():
    return ChatOpenAI(
        openai_api_key=HF_API_KEY,
        openai_api_base=HF_BASE_URL,
        model_name=MODEL_NAME,
        temperature=0.7,
        top_p=0.9,
        max_tokens=500,
        frequency_penalty=0.1,
        presence_penalty=0.0,
    )


def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def run_baseline_test(dataset_path, output_path, chain):
    try:
        df = pd.read_parquet(dataset_path)
        print(f"Loaded {len(df)} rows from {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    true_l1, true_l2 = [], []
    pred_l1, pred_l2 = [], []
    reasonings = []
    raw_outputs = []

    for index, row in df.iterrows():
        user_input = str(row['prompt'])

        real_l1 = str(row['L1_label']).strip().capitalize()
        real_l2 = str(row['community']).strip()

        if real_l1 == 'Safe':
            real_l2 = 'Safe_Community'

        true_l1.append(real_l1)
        true_l2.append(real_l2)

        input_data = {'question': user_input}
        print(f"Row {index + 1}/{len(df)} [Actual: {real_l1} | {real_l2}]", end="... ", flush=True)

        try:
            output = chain.invoke(input_data)
            raw_outputs.append(output)

            p_l1, p_l2, p_reasoning = "ERROR", "ERROR", ""

            for line in output.split('\n'):
                line = line.strip()
                if line.startswith('L1:'):
                    p_l1 = line.replace('L1:', '').strip().capitalize()
                elif line.startswith('L2:'):
                    p_l2 = line.replace('L2:', '').strip()
                elif line.startswith('Reasoning:'):
                    p_reasoning = line.replace('Reasoning:', '').strip()

            if p_l1 == "ERROR" or p_l2 == "ERROR":
                print(f"\n[DEBUG] Model did not follow the format. Raw output was:\n>>> {output} <<<\n")

            print(f"Prediction: {p_l1} | {p_l2}")
            time.sleep(1)

        except Exception as e:
            print(f"ERROR (API): {e}")
            p_l1, p_l2, p_reasoning = "ERROR", "ERROR", "ERROR"
            raw_outputs.append(f"ERROR: {e}")
            time.sleep(3)

        pred_l1.append(p_l1)
        pred_l2.append(p_l2)
        reasonings.append(p_reasoning)

    # ==========================================
    # 2. SCORE CALCULATION
    # ==========================================
    valid_indices = [i for i, p in enumerate(pred_l1) if p in ['Jailbreak', 'Safe']]

    print("\n" + "=" * 50)
    print(f"BASELINE TEST RESULTS: {MODEL_NAME}")
    print("=" * 50)

    if len(valid_indices) > 0:
        y_true_l1 = [true_l1[i] for i in valid_indices]
        y_pred_l1 = [pred_l1[i] for i in valid_indices]
        y_true_l2 = [true_l2[i] for i in valid_indices]
        y_pred_l2 = [pred_l2[i] for i in valid_indices]

        acc_l1 = accuracy_score(y_true_l1, y_pred_l1)
        f1_l1 = f1_score(y_true_l1, y_pred_l1, average='macro', zero_division=0)
        print("\n--- L1 (Jailbreak vs Safe) ---")
        print(f"Accuracy: {acc_l1:.4f}")
        print(f"Macro-F1: {f1_l1:.4f}")

        acc_l2 = accuracy_score(y_true_l2, y_pred_l2)
        f1_l2 = f1_score(y_true_l2, y_pred_l2, average='macro', zero_division=0)
        print("\n--- L2 (Attack Vectors) ---")
        print(f"Accuracy: {acc_l2:.4f}")
        print(f"Macro-F1: {f1_l2:.4f}")

        print(f"\nFailed/Malformed responses: {len(pred_l1) - len(valid_indices)}")
        print("\n--- L2 Detailed Report ---")
        print(classification_report(y_true_l2, y_pred_l2, zero_division=0))
    else:
        print("ERROR: No valid predictions could be made.")

    # ==========================================
    # 3. SAVE RESULTS (.parquet)
    # ==========================================
    df['Baseline_L1'] = pred_l1
    df['Baseline_L2'] = pred_l2
    df['Reasoning'] = reasonings
    df['Raw_Output'] = raw_outputs
    df.to_parquet(output_path)
    print(f"\nBaseline results saved to: {output_path}")


def main():
    llm = setup_llm_client()
    template_content = load_template("../../template2.pmt")
    prompt = PromptTemplate(input_variables=['question'], template=template_content)
    chain = prompt | llm | StrOutputParser()

    input_dataset = "test_dataset_split.parquet"
    output_dataset = f"baseline_{MODEL_NAME.split('/')[-1]}.parquet"

    run_baseline_test(input_dataset, output_dataset, chain)


if __name__ == "__main__":
    main()