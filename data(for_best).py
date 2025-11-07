# data.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
import random

# Fixed random seeds to ensure reproducible results
random.seed(42)

def load_and_split_data(file_path):

    # ========== Step 1: load data ==========
    df_info = pd.read_excel(file_path, sheet_name="LIst of PFAS in TSCA", usecols="A,C,J:O", header=0)
    df_info.columns = ["ID", "SMILES", "tox1", "tox2", "tox3", "tox4", "tox5", "tox6"]

    df_features = pd.read_excel(file_path, sheet_name="data after Zscore", usecols="B:PV", header=0)
    df = pd.concat([df_info, df_features], axis=1)

    # ========== Step 2: Generate molecular fingerprints ==========
    def smiles_to_fp(smiles, radius=2, nBits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)

    fps = [smiles_to_fp(s) for s in df["SMILES"]]
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]

    df = df.iloc[valid_idx].reset_index(drop=True)
    fps = [fps[i] for i in valid_idx]

    # ========== Step 3: Calculate distance matrix ==========
    def tanimoto_distance_matrix(fps):
        dists = []
        nfps = len(fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
        return dists

    dists = tanimoto_distance_matrix(fps)

    # ========== Step 4: Butina cluster ==========
    clusters = Butina.ClusterData(dists, len(fps), 0.5, isDistData=True)  # 保持原阈值 0.5

    # ========== Step 5: Divide the training set and testing set ==========
    train_ids, test_ids = [], []
    cluster_info = []

    for i, cluster in enumerate(clusters):
        cluster_mols = df.iloc[list(cluster)]["ID"].tolist()
        cluster_size = len(cluster_mols)
        cluster_info.append({"Cluster": i+1, "Molecules": cluster_mols, "Size": cluster_size})

        # Divide by cluster as a whole, with 80% allocated for training and 20% for testing, while maintaining the original logic
        if random.random() < 0.8:
            train_ids.extend(cluster_mols)
        else:
            test_ids.extend(cluster_mols)

    # ========== Step 6: output result ==========
    print("Number of molecules in the training set:", len(train_ids))
    print("Number of test set molecules:", len(test_ids))
    print("Overall proportion: training set {:.2f}, test set {:.2f}".format(
        len(train_ids)/len(df), len(test_ids)/len(df)
    ))

    print("\nExample of Molecular ID in Training Set:\n", train_ids[:20], "...")
    print("\nExample of Test Set Molecular ID:\n", test_ids[:20], "...")

    print("\nExamples of Cluster Information (Top 5):\n", cluster_info[:5])

    # ========== Step 7: return data ==========
    return df, train_ids, test_ids, cluster_info

# ========== main function ==========
if __name__ == "__main__":
    file_path = r"./Supplementary materials.xlsx"
    df, train_ids, test_ids, cluster_info = load_and_split_data(file_path)
