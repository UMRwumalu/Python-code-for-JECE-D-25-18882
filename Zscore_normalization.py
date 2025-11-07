import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# ------------------ Parameter Settings ------------------
file_path = r"./Supplementary materials.xlsx"
sheet_name = "data of high corr filtered"
# --------------------------------------------

# load data
df = pd.read_excel(file_path, sheet_name=sheet_name)
sample_names = df.iloc[:, 0]  # Sample Name
X = df.iloc[:, 1:]             # feature data

# Z-score normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# convert to DataFrame, Retain column names and sample names
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
final_df = pd.concat([sample_names, X_scaled_df], axis=1)

# Output normalized Excel
output_path = file_path.replace(".xlsx", "_Zscore_normalized.xlsx")
final_df.to_excel(output_path, index=False)
print("Normalization completed！")
print("Z-score normalized data is saved in：", output_path)

# ------------------ Save Normalizer ------------------
scaler_path = file_path.replace(".xlsx", "_scaler.pkl")
joblib.dump(scaler, scaler_path)
print("Z-score Standardized parameters have been saved as：", scaler_path)
