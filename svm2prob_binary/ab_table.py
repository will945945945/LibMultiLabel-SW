import pandas as pd
import os

def load_and_process_csv(folder, method_name):
    data_list = []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))

            required_columns = {"dataset", "model_type"}
            method_columns = {"platt": {"A", "B"}, "alpha_ce": {"alpha"}, "alpha_diff": {"alpha"},  "platt_onlyA": {"A"}}.get(method_name, set())
            df = df[df["mode"] == "trva"]
            df["method"] = method_name

            df = df[list(required_columns) + list(method_columns)]
            if method_name == "platt_onlyA":
                df = df.rename(columns={"A": "onlyA"})
            if method_name == "alpha_ce":
                df = df.rename(columns={"alpha": "alpha_ce"})
            if method_name == "alpha_diff":
                df = df.rename(columns={"alpha": "alpha_diff"})
            data_list.append(df)

    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()

def merge_methods(csvs):
    df = csvs[0]
    for i in csvs[1:]:
        df = pd.merge(df, i, on=["dataset", "model_type"], how="outer")
    return df

def format_latex_table(df):
    """
    Convert the merged dataframe into a LaTeX table.
    """
    df.sort_values(by=["dataset", "model_type"], inplace=True)

    latex_lines = []
    last_dataset = None
    last_model = None

    for _, row in df.iterrows():
        dataset = row["dataset"]
        model_type = row["model_type"]
        if model_type == "lr":
            continue
        alpha_ce = row.get("alpha_diff", "-")
        alpha_diff = row.get("alpha_ce", "-")
        A = row.get("A", "-")
        B = row.get("B", "-")
        onlyA = row.get("onlyA", "-")

        dataset_str = dataset if dataset != last_dataset else ""
        if dataset_str != "":
            latex_lines.append("\\hline")

        model_str = model_type if (dataset != last_dataset or model_type != last_model) else ""
        if model_str != "" and dataset_str == "":
            latex_lines.append("\\cline{2-7}")

        latex_lines.append(f"{dataset_str} & {model_str} & {alpha_ce:.2f} & {alpha_diff:.2f} & {abs(onlyA):.2f} & {abs(A):.2f} & {B:.2f} \\\\")

        last_dataset = dataset
        last_model = model_type

    # Create LaTeX table structure
    column_format = "llccccc"
    header = "Dataset & Model Type & Alpha_CE & Alpha_Diff & $\\mid\\text{onlyA}\\mid$ & $\\mid\\text{A}\\mid$ & B \\\\"
    latex_table = "\\begin{tabular}{" + column_format + "}\n" + header + "\n"
    latex_table += "\n".join(latex_lines)
    latex_table += "\n\\hline\n\\end{tabular}"
    latex_table = latex_table.replace("_", "\_")
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
    if os.path.basename(i) == "franc":
        continue
    csvs.append(load_and_process_csv(i, os.path.basename(i)))

# Merge methods based on dataset and model_type
df = merge_methods(csvs)
# Generate LaTeX table
latex_table = format_latex_table(df)

# Print and save
print(latex_table)
os.makedirs(f"tables", exist_ok=True)
if root == "no_tune":
    with open(f"tables/ab_table_no_tune.tex", "w") as f:
        f.write(latex_table)
else:
    with open(f"tables/ab_table_tune.tex", "w") as f:
        f.write(latex_table)
