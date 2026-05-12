import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

def generate_hcse_excel(parquet_file, excel_file):
    print(f"Reading data from: {parquet_file}...")
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Standardize values just in case
    df['HCSE_L1'] = df['HCSE_L1'].fillna('ERROR').astype(str).str.strip().str.capitalize()
    df['L1_label'] = df['L1_label'].fillna('ERROR').astype(str).str.strip().str.capitalize()

    # Count failed (ERROR) queries
    failed_l1 = (df['HCSE_L1'] == 'ERROR').sum() + (df['HCSE_L1'] == 'ERROR').sum()

    # Keep only successful queries for metric calculation
    valid_mask = df['HCSE_L1'].isin(['Jailbreak', 'Safe'])
    df_valid = df[valid_mask].copy()

    if df_valid.empty:
        print("Error: No successful L1 classifications found in the dataset.")
        return

    # ==========================================
    # L1 LEVEL METRICS (Safe / Adversarial)
    # ==========================================
    l1_true = df_valid['L1_label']
    l1_pred = df_valid['HCSE_L1']

    l1_accuracy = accuracy_score(l1_true, l1_pred)
    l1_macro_f1 = f1_score(l1_true, l1_pred, average='macro', zero_division=0)

    # Specific error types
    jailbreak_as_safe = sum((l1_true == 'Jailbreak') & (l1_pred == 'Safe'))
    safe_as_jailbreak = sum((l1_true == 'Safe') & (l1_pred == 'Jailbreak'))

    # ==========================================
    # L2 LEVEL METRICS (Attack Communities)
    # ==========================================
    l2_true = []
    for index, row in df_valid.iterrows():
        real_l1 = str(row['L1_label']).strip().capitalize()
        if real_l1 == 'Safe':
            l2_true.append('Safe_Community')
        else:
            l2_true.append(str(row['community']).strip())

    l2_pred = df_valid['HCSE_L2'].fillna('ERROR').astype(str).str.strip()

    l2_accuracy = accuracy_score(l2_true, l2_pred)
    l2_macro_f1 = f1_score(l2_true, l2_pred, average='macro', zero_division=0)

    # Detailed L2 report
    l2_report_dict = classification_report(l2_true, l2_pred, output_dict=True, zero_division=0)
    df_l2_report = pd.DataFrame(l2_report_dict).transpose()

    df_l2_report = df_l2_report.rename(columns={
        'precision': 'Precision',
        'recall': 'Recall',
        'f1-score': 'F1-Score',
        'support': 'Support'
    })
    df_l2_report.index.name = 'Attack Community'
    df_l2_report = df_l2_report.reset_index()

    # ==========================================
    # BUILD SUMMARY TABLE
    # ==========================================
    summary_data = {
        'System': ['HCSE Ensemble'],
        'L1_Accuracy': [round(l1_accuracy, 4)],
        'L1_Macro-F1': [round(l1_macro_f1, 4)],
        'Missed Attacks (JB -> Safe)': [jailbreak_as_safe],
        'False Alarms (Safe -> JB)': [safe_as_jailbreak],
        'L2_Accuracy': [round(l2_accuracy, 4)],
        'L2_Macro-F1': [round(l2_macro_f1, 4)],
        'Failed (ERROR)': [failed_l1]
    }
    df_summary = pd.DataFrame(summary_data)

    # ==========================================
    # SAVE TO EXCEL
    # ==========================================
    print(f"Saving tables to: {excel_file}...")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_l2_report.to_excel(writer, sheet_name='L2_Community_Report', index=False)

    print("Done! Excel file created successfully.")

if __name__ == "__main__":
    input_file = 'hcse_final_results_zero_shot.parquet'
    output_file = 'hcse_final_results.xlsx'

    generate_hcse_excel(input_file, output_file)