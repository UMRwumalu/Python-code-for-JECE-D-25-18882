import pandas as pd
import matplotlib.pyplot as plt

# ==================== load data ====================
file_path = r"./Supplementary materials.xlsx"

# laod descriptors sheet（No header set, manually handle）
df = pd.read_excel(file_path, sheet_name="descriptors", header=None)

# Feature Name: Column 2 to 1445 in the first row
feature_names = df.iloc[0, 1:].values

# Sample Name: Column 1, Line 2 to Line 269
sample_names = df.iloc[1:, 0].values

# Feature matrix: rows 2 to 269, columns 2 to 1445
X = df.iloc[1:, 1:].astype(float)
X.columns = feature_names
X.index = sample_names

# ==================== Zero variance filtering ====================
# Calculate the variance of each feature
variances = X.var(axis=0)

# Identify features with variance of 0
removed_features = variances[variances == 0].index
kept_features = variances[variances != 0].index

print("Number of reserved features：", len(kept_features))
print("Number of deleted zero variance features：", len(removed_features))
print("Deleted zero variance features：")
for f in removed_features:
    print(f)

# Filter out zero variance features
X_cleaned = X[kept_features]

# ==================== Plot: Distribution of characteristic variance ====================
plt.figure(figsize=(8,5))
plt.hist(variances, bins=50, color="skyblue", edgecolor="black")
plt.axvline(x=0.0, color="red", linestyle="--", label="Zero Variance")
plt.xlabel("Feature Variance", fontsize=12)
plt.ylabel("Number of Features", fontsize=12)
plt.title("Variance Distribution of Features", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

# ==================== Save the cleaned data ====================
# Splicing back the sample list
cleaned_df = pd.concat([pd.Series(sample_names, name="Sample"), X_cleaned], axis=1)

# 固定输出目录
output_path = r"C:/Users/*****/Desktop/******/cleaned_zero_variance.xlsx"
cleaned_df.to_excel(output_path, index=False)

# save the list of deleted features
removed_path = r"C:/Users/******/Desktop/*******/removed_zero_variance_features.xlsx"
pd.DataFrame(removed_features, columns=["Removed_Features"]).to_excel(removed_path, index=False)

print(f"\nThe cleaned data has been saved to：{output_path}")
print(f"The deleted feature list has been saved to：{removed_path}")
