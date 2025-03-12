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
            df = df[["dataset", "model_type", "method", "tr_NLL", "te_NLL"]]
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
        if model_type != "lr":
            continue
        lr.append([tr_nll, te_nll])

    cnt = 0
    for _, row in df.iterrows():
        dataset = row["dataset"]
        model_type = row["model_type"]
        method = row["method"] if model_type != "lr" else ""
        if method == "Prob_lr":
            method = "Alpha"
        tr_nll = row["tr_NLL"]
        te_nll = row["te_NLL"]

        # Only show dataset name if it has changed
        dataset_str = dataset if dataset != last_dataset else ""
        if dataset_str != "" and cnt != 0:
            latex_lines.append("\\hline")
        # Only show model type if it has changed within the dataset
        model_str = model_type if (dataset != last_dataset or model_type != last_model) else ""
        if model_str != "" and dataset_str == "":
            latex_lines.append("\\cline{2-5}")
        
        rel_tr_nll =  tr_nll - lr[cnt][0]
        rel_tr_ratio = rel_tr_nll / lr[cnt][0]
        rel_te_nll =  te_nll - lr[cnt][1]
        rel_te_ratio = rel_te_nll / lr[cnt][1]

        if model_type == "lr":
            cnt += 1
            latex_lines.append(
            f"{dataset_str} & {model_str} & {method} & {tr_nll:.4f} & {te_nll:.4f} \\\\"
        )
        else:
            sign_tr = "$+$" if rel_tr_nll > 0 else "$-$"
            sign_te = "$+$" if rel_te_nll > 0 else "$-$"
            latex_lines.append(
            f"{dataset_str} & {model_str} & {method} & {tr_nll:.4f}({sign_tr}{abs(rel_tr_ratio)*100:.2f}\%) & {te_nll:.4f}({sign_te}{abs(rel_te_ratio)*100:.2f}\%) \\\\"
        )
        
        last_dataset = dataset
        last_model = model_type

    # Create the complete LaTeX table code
    column_format = "@{}lllcc@{}"
    header = "Dataset & Model Type & Method & Train_CE & Test_CE \\\\" + "\n" +"\\hline"
    latex_table = "\\begin{tabular}{" + column_format + "}\n" + header + "\n"
    latex_table += "\n".join(latex_lines)
    latex_table += "\n\\hline\n\\end{tabular}"
    latex_table = latex_table.replace("_", "\\_")
    return latex_table

# Define the paths to your CSV folders for the two methods
folder1 = "tune/platt"  # Replace with your actual path
folder2 = "tune/alpha"  # Replace with your actual path
folder3 = "tune/franc"
folder4 = "tune/platt_A"

# Load and process the CSV data for each method
df_method1 = load_and_process_csv(folder1, "Platt")
df_method2 = load_and_process_csv(folder2, "Alpha")
df_method3 = load_and_process_csv(folder3, "Franc")
df_method4 = load_and_process_csv(folder4, "Platt_A")

# Combine the data from both methods
final_df = pd.concat([df_method1, df_method2, df_method3, df_method4])
# Generate the LaTeX table
latex_table = format_latex_table(final_df)

# Print and save the LaTeX table code
print(latex_table)
with open("ce_table_tune.tex", "w") as f:
    f.write(latex_table)
