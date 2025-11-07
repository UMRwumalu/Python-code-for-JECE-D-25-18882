import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Parameter Settings ------------------
file_path = r"./Supplementary materials.xlsx"
feature_sheet = "data after Zscore"
output_sheet = "LIst of PFAS in TSCA"
output_fig_path = file_path.replace(".xlsx", "_UMAP_uniform_color.png")
# --------------------------------------------

# Set the font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Read feature data (columns 2 to 480)
feature_df = pd.read_excel(file_path, sheet_name=feature_sheet)
X = feature_df.iloc[:, 1:480] 

# Read output data (columns J~O, Python indexes 9~14)
output_df = pd.read_excel(file_path, sheet_name=output_sheet)
Y = output_df.iloc[:, 9:15]

# Ensure consistent number of rows
assert len(X) == len(Y), "Inconsistent number of features and output lines！"

# Use UMAP for dimensionality reduction, adjust parameters to make point distribution more uniform
reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=42)  # Increase the number of neighbors and minimum distance
X_umap = reducer.fit_transform(X)

# convert to DataFrame And merge the output values
umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
umap_df = pd.concat([umap_df, Y.reset_index(drop=True)], axis=1)

# drawing
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(Y.columns):
    sc = axes[i].scatter(
        umap_df["UMAP1"], umap_df["UMAP2"],
        c=umap_df[col], cmap="viridis_r", s=50, alpha=0.8  
    )
    axes[i].set_title(col, fontsize=14)
    axes[i].set_xlabel("UMAP1", fontsize=12)
    axes[i].set_ylabel("UMAP2", fontsize=12)
    cbar = plt.colorbar(sc, ax=axes[i])
    cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(output_fig_path, dpi=300)
plt.show()
