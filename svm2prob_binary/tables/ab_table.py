import pandas as pd
import os

def load_and_process_csv(folder, method_name):
    data_list = []

    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))

            required_columns = {"dataset", "model_type"}
            method_columns = {"Platt": {"A", "B"}, "Alpha": {"best_alpha"}}.get(method_name, set())
            df = df[df["mode"] == "trva"]
            df["method"] = method_name

            df = df[list(required_columns) + list(method_columns)]
            
            data_list.append(df)

    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()

def merge_methods(df_platt, df_prob_lr, df_platt_A):
    return pd.merge(df_prob_lr, df_platt, on=["dataset", "model_type"], how="outer")

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
        best_alpha = row.get("best_alpha", "-")
        A = row.get("A", "-")
        B = row.get("B", "-")

        dataset_str = dataset if dataset != last_dataset else ""
        if dataset_str != "":
            latex_lines.append("\\hline")

        model_str = model_type if (dataset != last_dataset or model_type != last_model) else ""
        if model_str != "" and dataset_str == "":
            latex_lines.append("\\cline{2-5}")

        latex_lines.append(f"{dataset_str} & {model_str} & {best_alpha:.2f} & {abs(A):.2f} & {B:.2f} \\\\")

        last_dataset = dataset
        last_model = model_type

    # Create LaTeX table structure
    column_format = "llccc"
    header = "Dataset & Model Type & Best Alpha & |A| & B \\\\"
    latex_table = "\\begin{tabular}{" + column_format + "}\n" + header + "\n"
    latex_table += "\n".join(latex_lines)
    latex_table += "\n\\hline\n\\end{tabular}"

    return latex_table

# Define paths for the two methods
folder_platt = "no_tune/platt"
folder_prob_lr = "no_tune/alpha"
folder_platt_A = "no_tune/platt_A"

# Load CSVs
df_platt = load_and_process_csv(folder_platt, "Platt")
df_prob_lr = load_and_process_csv(folder_prob_lr, "Alpha")
df_platt_A = load_and_process_csv(folder_platt_A, "Platt_A")
# Merge methods based on dataset and model_type
final_df = merge_methods(df_platt, df_prob_lr, df_platt_A)

# Generate LaTeX table
latex_table = format_latex_table(final_df)

# Print and save
print(latex_table)
with open("ab_table_no_tune.tex", "w") as f:
    f.write(latex_table)
