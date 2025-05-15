import pandas as pd
import os

def load_and_process_csv(folder, method_name):
    """
    Load all CSV files in a folder, extract the relevant columns,
    filter rows where mode=='trva', and add a method identifier.
    """
    data_list = []
    
    for file in os.listdir(folder):
        if os.path.basename(file) != "rcv1.csv" and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            df = df[df["mode"] == "trva"]
            df["method"] = method_name if method_name != "alpha_ce" else "Ours"
            df = df[["dataset", "model_type", "method", "te_NLL"]]
            if method_name != "alpha":
                df = df[df["model_type"] != "LR"]
            data_list.append(df)
    
    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()

def format_latex_table(df1, df2):
    df1.sort_values(by=["dataset", "model_type", "method"], inplace=True)
    df2.sort_values(by=["dataset", "model_type", "method"], inplace=True)

    latex_lines = []
    last_dataset = None
    last_model = None
    lr = []
    for (_, row1), (__, row2) in zip(df1.iterrows(), df2.iterrows()):
        dataset = row1["dataset"] if row1["dataset"] != "rcv1_reverse" else "rcv1"
        model_type = row1["model_type"]
        method = row1["method"]
        tr_nll_1 = row1["tr_NLL"]
        te_nll_1 = row1["te_NLL"]
        tr_nll_2 = row2["tr_NLL"]
        te_nll_2 = row2["te_NLL"]
        if model_type != "LR":
            continue
        lr.append([tr_nll_1, te_nll_1, tr_nll_2, te_nll_2])

    cnt = 0
    for (_, row1), (__, row2) in zip(df1.iterrows(), df2.iterrows()):
        dataset = row1["dataset"]
        model_type = row1["model_type"]
        method = row1["method"].title() if model_type != "LR" else ""
        tr_nll_1 = row1["tr_NLL"]
        te_nll_1 = row1["te_NLL"]
        tr_nll_2 = row2["tr_NLL"]
        te_nll_2 = row2["te_NLL"]
        # tr_diff = row["tr_diff"]
        # te_diff = row["te_diff"]

        dataset_str = dataset if dataset != last_dataset else ""
        if dataset_str != "" and cnt != 0:
            latex_lines.append("\\midrule")
        model_str = model_type if (dataset != last_dataset or model_type != last_model) else ""
        if model_str != "" and dataset_str == "":
            latex_lines.append("\\cmidrule{2-11}")

        rel_tr_nll_1 = tr_nll_1 - lr[cnt][0]
        rel_tr_ratio_1 = rel_tr_nll_1 / lr[cnt][0]
        rel_te_nll_1 = te_nll_1 - lr[cnt][1]
        rel_te_ratio_1 = rel_te_nll_1 / lr[cnt][1]

        rel_tr_nll_2 = tr_nll_2 - lr[cnt][2]
        rel_tr_ratio_2 = rel_tr_nll_2 / lr[cnt][2]
        rel_te_nll_2 = te_nll_2 - lr[cnt][3]
        rel_te_ratio_2 = rel_te_nll_2 / lr[cnt][3]

        if model_type == "LR":
            cnt += 1
            latex_lines.append(
                f"{dataset_str} & {model_str} & {method} & {tr_nll_1:.4f} & & {te_nll_1:.4f} & & {tr_nll_2:.4f} & & {te_nll_2:.4f} \\\\"
            )
            # latex_lines.append(
            #     f"{dataset_str} & {model_str} & {method} & {tr_nll:.4f} & {te_nll:.4f} & \\\\"
            # )
        else:
            sign_tr_1 = "$+$" if rel_tr_nll_1 > 0 else "$-$"
            sign_te_1 = "$+$" if rel_te_nll_1 > 0 else "$-$"
            sign_tr_2 = "$+$" if rel_tr_nll_2 > 0 else "$-$"
            sign_te_2 = "$+$" if rel_te_nll_2 > 0 else "$-$"
            latex_lines.append(
                f"{dataset_str} & {model_str} & {method} & "
                f"{tr_nll_1:.4f} & {sign_tr_1}{abs(rel_tr_ratio_1)*100:.2f}\\% & "
                f"{te_nll_1:.4f} & {sign_te_1}{abs(rel_te_ratio_1)*100:.2f}\\% & "
                f"{tr_nll_2:.4f} & {sign_tr_2}{abs(rel_tr_ratio_2)*100:.2f}\\% & "
                f"{te_nll_2:.4f} & {sign_te_2}{abs(rel_te_ratio_2)*100:.2f}\\% \\\\"
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
    column_format = "@{}lll c@{ }rc@{ }r c@{ }rc@{ }r@{}"
    header = (
        "\\multirow{2}{*}{Dataset} & Model & \\multirow{2}{*}{Method} & \\multicolumn{4}{c}{Tuned} & \\multicolumn{4}{c}{Untuned} \\\\"
        "& Type &  & \\multicolumn{2}{c}{Train_CE} & \\multicolumn{2}{c}{Test_CE} & \\multicolumn{2}{c}{Train_CE} & \\multicolumn{2}{c}{Test_CE} \\\\"
        "\n\\toprule"
    )
    latex_table = "\\begin{tabular}{" + column_format + "}\n" + header + "\n"
    latex_table += "\n".join(latex_lines)
    latex_table += "\n\\bottomrule\n\\end{tabular}"
    latex_table = latex_table.replace("_", "\\_")
    return latex_table

folders = ["tables/no_tune/*", "tables/tune/*"]
from glob import glob
folders = [sorted(glob(folder)) for folder in folders]
csvs = []
for folder in folders:
    csv = []
    for i in folder:
        if os.path.basename(i) in ["alpha_diff", "platt_onlyA", "liblinear", "liblinear_2", "HFY"]:
            continue
        csv.append(load_and_process_csv(i, os.path.basename(i)))
    csvs.append(csv)

dfs = [pd.concat(csv) for csv in csvs]
latex_table = format_latex_table(dfs[0], dfs[1])

print(latex_table)
os.makedirs(f"tables", exist_ok=True)
with open(f"tables/ce_table.tex", "w") as f:
    f.write(latex_table)