import glob
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compare_baseline_results():
    # 1. Find all files that start with "baseline_" and end with ".parquet"
    files = glob.glob("baseline_*.parquet")

    if not files:
        print("No baseline_*.parquet files found!")
        return

    results = []
    detailed_reports = {}
    all_detailed_stats_for_excel = []  # NEW: Store all detailed stats for the Excel sheet

    print(f"Found {len(files)} baseline files. Calculating summary and Excel tables...\n")

    # 2. Iterate through each file
    for file in files:
        model_name = file.replace("baseline_", "").replace(".parquet", "").replace(".parquet", "")

        try:
            df = pd.read_parquet(file)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        # Load ground truth labels
        true_l1 = df['L1_label'].astype(str).str.strip().str.capitalize().tolist()
        true_l2 = df['community'].astype(str).str.strip().tolist()

        # Fix L2 Safe_Community logic
        true_l2 = ['Safe_Community' if l1 == 'Safe' else l2 for l1, l2 in zip(true_l1, true_l2)]

        # Load model predictions
        pred_l1 = df['Baseline_L1'].astype(str).str.strip().tolist()
        pred_l2 = df['Baseline_L2'].astype(str).str.strip().tolist()

        # Filter out invalid formatting errors from the model
        valid_indices = [i for i, p in enumerate(pred_l1) if p in ['Jailbreak', 'Safe']]

        y_true_l1 = [true_l1[i] for i in valid_indices]
        y_pred_l1 = [pred_l1[i] for i in valid_indices]

        y_true_l2 = [true_l2[i] for i in valid_indices]
        y_pred_l2 = [pred_l2[i] for i in valid_indices]

        # 3. Calculate metrics and compile the detailed hit-rate report
        if valid_indices:
            acc_l1 = accuracy_score(y_true_l1, y_pred_l1)
            f1_l1 = f1_score(y_true_l1, y_pred_l1, average='macro', zero_division=0)

            # --- UUS LOOGIKA: L1 Vigade jaotus ---
            jailbreak_as_safe = sum(1 for yt, yp in zip(y_true_l1, y_pred_l1) if yt == 'Jailbreak' and yp == 'Safe')
            safe_as_jailbreak = sum(1 for yt, yp in zip(y_true_l1, y_pred_l1) if yt == 'Safe' and yp == 'Jailbreak')

            acc_l2 = accuracy_score(y_true_l2, y_pred_l2)
            f1_l2 = f1_score(y_true_l2, y_pred_l2, average='macro', zero_division=0)

            unique_classes = sorted(list(set(y_true_l2)))
            class_stats_for_print = []

            for cls in unique_classes:
                total_in_data = y_true_l2.count(cls)
                correct_preds = sum(1 for yt, yp in zip(y_true_l2, y_pred_l2) if yt == cls and yp == cls)
                missed_preds = total_in_data - correct_preds
                hit_rate = (correct_preds / total_in_data) * 100 if total_in_data > 0 else 0.0

                # Dictionary for terminal print (without model name to save space)
                class_stats_for_print.append({
                    "Category (Community)": cls,
                    "Total (Support)": total_in_data,
                    "Correct Hits": correct_preds,
                    "Missed (Errors)": missed_preds,
                    "Hit Rate (%)": f"{hit_rate:.1f}%"
                })

                # Dictionary for Excel (includes model name so we can filter in Excel)
                all_detailed_stats_for_excel.append({
                    "Model": model_name,
                    "Category (Community)": cls,
                    "Total (Support)": total_in_data,
                    "Correct Hits": correct_preds,
                    "Missed (Errors)": missed_preds,
                    "Hit Rate (%)": round(hit_rate, 1)  # Keeping as float for Excel charts
                })

            # Format the detailed category stats into a Markdown table for the terminal
            report_df = pd.DataFrame(class_stats_for_print)
            detailed_reports[model_name] = report_df.to_markdown(index=False)

        else:
            acc_l1, f1_l1, acc_l2, f1_l2 = 0, 0, 0, 0
            jailbreak_as_safe, safe_as_jailbreak = 0, 0
            detailed_reports[model_name] = "ERROR: No valid predictions found."

        errors = len(pred_l1) - len(valid_indices)

        # Append to the main summary list (koos uute veergudega)
        results.append({
            "Model": model_name,
            "L1_Accuracy": float(acc_l1),
            "L1_Macro-F1": float(f1_l1),
            "JB_as_Safe(Missed)": jailbreak_as_safe,
            "Safe_as_JB(FalseAlarm)": safe_as_jailbreak,
            "L2_Accuracy": float(acc_l2),
            "L2_Macro-F1": float(f1_l2),
            "Failed Responses": errors
        })

    # 4. Format the summary results using Pandas DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="L2_Macro-F1", ascending=False)

    # Convert floats to nicely formatted strings for the terminal printout
    print_df = results_df.copy()
    for col in ["L1_Accuracy", "L1_Macro-F1", "L2_Accuracy", "L2_Macro-F1"]:
        print_df[col] = print_df[col].apply(lambda x: f"{x:.4f}")

    print("===========================================================================================")
    print("                               BASELINE MODEL COMPARISON                                   ")
    print("===========================================================================================")
    print(print_df.to_markdown(index=False))
    print("===========================================================================================\n")

    print("=========================================================================")
    print("             DETAILED L2 (COMMUNITY) HIT RATE BY MODEL                   ")
    print("=========================================================================")
    for idx, row in results_df.iterrows():
        model_name = row['Model']
        print(f"\n--- MODEL: {model_name} ---")
        print(detailed_reports.get(model_name, "No data available."))
        print("-" * 65)

    # 5. EXCEL EXPORT
    excel_filename = "../../baseline_comparison_results.xlsx"
    detailed_excel_df = pd.DataFrame(all_detailed_stats_for_excel)

    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name="Summary", index=False)
            detailed_excel_df.to_excel(writer, sheet_name="Detailed_Hit_Rates", index=False)
        print(f"\n✅ SUCCESS: All data has been saved to '{excel_filename}'")
        print("   -> Sheet 1: 'Summary' (Overall macro scores, plus L1 confusion metrics)")
        print("   -> Sheet 2: 'Detailed_Hit_Rates' (Category breakdowns)")
    except Exception as e:
        print(f"\n❌ ERROR saving to Excel: {e}")

if __name__ == "__main__":
    compare_baseline_results()