import pandas as pd
import numpy as np

# ------------------ Parameter Settings ------------------
file_path = r"./Supplementary materials.xlsx"
sheet_name = "cleaned zero variance"
threshold = 0.95 
# --------------------------------------------

# load data
df = pd.read_excel(file_path, sheet_name=sheet_name)
sample_names = df.iloc[:, 0]
X = df.iloc[:, 1:]

removed_features = []  # Excluded feature list
removed_info = []      # Correlation information of removed features

# Calculate Spearman correlation coefficient matrix
corr_matrix = X.corr(method='spearman').abs()

cols = X.columns.tolist()
keep_features = set(cols)  # Current reserved feature set

# Traverse the upper triangular matrix
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        f1 = cols[i]
        f2 = cols[j]
        corr_value = corr_matrix.loc[f1, f2]
        if corr_value >= threshold:
          #If f1 is in the reserved set, then remove f1
            if f1 in keep_features:
                keep_features.remove(f1)
                removed_features.append(f1)
                removed_info.append({"Removed Feature": f1,
                                     "Reason Feature": f2,
                                     "Correlation": corr_value})

# Build retained data
X_filtered = X[list(keep_features)]

# Output preserved feature data
output_path = file_path.replace(".xlsx", "_high_corr_filtered.xlsx")
final_df = pd.concat([sample_names, X_filtered], axis=1)
final_df.to_excel(output_path, index=False)

# Output the reason table for feature exclusion
removed_info_df = pd.DataFrame(removed_info)
removed_info_path = file_path.replace(".xlsx", "_removed_features_info.xlsx")
removed_info_df.to_excel(removed_info_path, index=False)

# Console output information
print("Number of excluded features：", len(removed_features))
print("Retain the number of features：", len(keep_features))
print("Process completed！")
print("The cleaned data is saved in：", output_path)
print("Excluded feature information is saved in：", removed_info_path)
