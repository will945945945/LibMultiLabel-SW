import pandas as pd
import os

def load_and_process_csv(folder, method_name):
    """
    Load all CSV files in a folder, extract the relevant columns,
    filter rows where mode=='trva', and add a method identifier.
    """
    data_list = []
    
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            df = df[df["mode"] == "trva"]
            df["method"] = method_name
            df = df[["dataset", "model_type", "method", "tr_NLL", "te_NLL", "tr_Acc", "te_Acc", "tr_diff", "te_diff"]]
            if method_name != "alpha_ce":
                df = df[df["model_type"] != "lr"]
            data_list.append(df)
    
    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()

def format_latex_table(df):
    df.sort_values(by=["dataset", "model_type", "method"], inplace=True)

    latex_lines = []
    last_dataset = None
    last_model = None
    lr = []
    for _, row in df.iterrows():
        dataset = row["dataset"]
        model_type = row["model_type"]
        method = row["method"]
        tr_nll = row["tr_NLL"]
        te_nll = row["te_NLL"]
        tr_acc = row["tr_Acc"]
        te_acc = row["te_Acc"]
        if model_type != "lr":
            continue
        lr.append([tr_nll, te_nll, tr_acc, te_acc])

    cnt = 0
    for _, row in df.iterrows():
        dataset = row["dataset"]
        model_type = row["model_type"]
        method = row["method"] if model_type != "lr" else ""
        tr_nll = row["tr_NLL"]
        te_nll = row["te_NLL"]
        tr_acc = row["tr_Acc"]
        te_acc = row["te_Acc"]
        tr_diff = row["tr_diff"]
        te_diff = row["te_diff"]

        dataset_str = dataset if dataset != last_dataset else ""
        if dataset_str != "" and cnt != 0:
            latex_lines.append("\\midrule")
        model_str = model_type if (dataset != last_dataset or model_type != last_model) else ""
        if model_str != "" and dataset_str == "":
            latex_lines.append("\\cmidrule{2-9}")

        rel_tr_nll = tr_nll - lr[cnt][0]
        rel_tr_ratio = rel_tr_nll / lr[cnt][0]
        rel_te_nll = te_nll - lr[cnt][1]
        rel_te_ratio = rel_te_nll / lr[cnt][1]

        rel_tr_acc = tr_acc - lr[cnt][2]
        rel_tr_acc_ratio = rel_tr_acc / lr[cnt][2]
        rel_te_acc = te_acc - lr[cnt][3]
        rel_te_acc_ratio = rel_te_acc / lr[cnt][3]

        if model_type == "lr":
            cnt += 1
            latex_lines.append(
                f"{dataset_str} & {model_str} & {method} & {tr_nll:.4f} & {te_nll:.4f} & {tr_acc:.4f} & {te_acc:.4f} & \\\\"
            )
            # latex_lines.append(
            #     f"{dataset_str} & {model_str} & {method} & {tr_nll:.4f} & {te_nll:.4f} & \\\\"
            # )
        else:
            sign_tr = "$+$" if rel_tr_nll > 0 else "$-$"
            sign_te = "$+$" if rel_te_nll > 0 else "$-$"
            sign_tr_acc = "$+$" if rel_tr_acc > 0 else "$-$"
            sign_te_acc = "$+$" if rel_te_acc > 0 else "$-$"
            latex_lines.append(
                f"{dataset_str} & {model_str} & {method} & "
                f"{tr_nll:.4f}({sign_tr}{abs(rel_tr_ratio)*100:.2f}\\%) & "
                f"{te_nll:.4f}({sign_te}{abs(rel_te_ratio)*100:.2f}\\%) & "
                f"{tr_acc:.4f}({sign_tr_acc}{abs(rel_tr_acc_ratio)*100:.2f}\\%) & "
                f"{te_acc:.4f}({sign_te_acc}{abs(rel_te_acc_ratio)*100:.2f}\\%) & "
                f"{tr_diff:.2f} & {te_diff:.2f} \\\\"
            )
            # latex_lines.append(
            #     f"{dataset_str} & {model_str} & {method} & "
            #     f"{tr_nll:.4f}({sign_tr}{abs(rel_tr_ratio)*100:.2f}\\%) & "
            #     f"{te_nll:.4f}({sign_te}{abs(rel_te_ratio)*100:.2f}\\%) & "
            #     f"{tr_diff:.2f} & {te_diff:.2f} \\\\"
            # )

        last_dataset = dataset
        last_model = model_type

    # Assemble the final table
    column_format = "@{}lllcccccc@{}"
    header = (
        "Dataset & Model Type & Method & Train_CE & Test_CE & Train_Acc & Test_Acc & Train_Diff & Test_Diff \\\\"
        # "Dataset & Model Type & Method & Train_CE & Test_CE & Train_Diff & Test_Diff \\\\"
        + "\n\\toprule"
    )
    latex_table = "\\begin{tabular}{" + column_format + "}\n" + header + "\n"
    latex_table += "\n".join(latex_lines)
    latex_table += "\n\\bottomrule\n\\end{tabular}"
    latex_table = latex_table.replace("_", "\\_")
    return latex_table

import sys
root = sys.argv[1]
# Define paths for the two methods
folders = f"tables/{root}/*"
from glob import glob
folders = sorted(glob(folders))
# Load CSVs
csvs = []
for i in folders:

    if os.path.basename(i) in ["alpha_diff", "platt_onlyA", "liblinear", "liblinear_2", "HFY"]:
        continue
    csvs.append(load_and_process_csv(i, os.path.basename(i)))

# Combine the data from both methods
final_df = pd.concat(csvs)
# Generate the LaTeX table
latex_table = format_latex_table(final_df)

# Print and save
print(latex_table)
os.makedirs(f"tables", exist_ok=True)
if root == "no_tune":
    with open(f"tables/ce_table_no_tune.tex", "w") as f:
        f.write(latex_table)
else:
    with open(f"tables/ce_table_tune.tex", "w") as f:
        f.write(latex_table)