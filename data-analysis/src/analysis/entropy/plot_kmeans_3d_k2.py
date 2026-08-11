import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os 

# ======================================================================
# CONFIGURATION DES CHEMINS
# ======================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Le script se trouve dans src/analysis/entropy/.
BASE_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "..",
    )
)

HUMAN_DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Ragni2016.csv",
)

MODELS_FILE = os.path.join(
    BASE_DIR,
    "results",
    "tables",
    "mental_models",
    "mental_models_count.csv",
)

OUTPUT_DIRECTORY = os.path.join(
    BASE_DIR,
    "results",
    "analysis",
    "entropy",
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "kmeans_3d_2k.png",
)


os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


VALID_CONCLUSIONS = {
    'AA1': 'Aac', 'EA1': 'Eac', 'AE2': 'Eac', 'EA2': 'Eac',
    'AA3': 'Iac', 'IA3': 'Iac', 'AI3': 'Iac', 'EA3': 'Oac',
    'OA3': 'Oac', 'EI3': 'Oac', 'AA4': 'Ica', 'AE4': 'Eac',
    'IA4': 'Ica', 'EA4': 'Oac', 'EI4': 'Oac'
}

def encode_syllogism(row):
    task_str, resp_str = row['task'], row['response']
    prem1, prem2 = task_str.split('/')
    q1, t1_1, t1_2 = prem1.split(';')
    q2, t2_1, t2_2 = prem2.split(';')
    
    quant_map = {'All': 'A', 'Some': 'I', 'No': 'E', 'Some not': 'O'}
    quant1, quant2 = quant_map[q1], quant_map[q2]
    
    if t1_2 == t2_1: figure, A, C = '1', t1_1, t2_2
    elif t1_1 == t2_2: figure, A, C = '2', t1_2, t2_1
    elif t1_2 == t2_2: figure, A, C = '3', t1_1, t2_1
    elif t1_1 == t2_1: figure, A, C = '4', t1_2, t2_2
    else: raise ValueError("Figure inconnue")
        
    task_enc = f"{quant1}{quant2}{figure}"
    
    if resp_str == 'NVC':
        resp_enc = 'NVC'
    else:
        r_q, r_t1, r_t2 = resp_str.split(';')
        r_quant = quant_map[r_q]
        if r_t1 == A and r_t2 == C: direction = 'ac'
        elif r_t1 == C and r_t2 == A: direction = 'ca'
        else: direction = '??'
        resp_enc = f"{r_quant}{direction}"
        
    return pd.Series([task_enc, resp_enc], index=['task', 'response_enc'])

def main():
    print("1. Traitement des données...")
    try:
        df_human = pd.read_csv(HUMAN_DATA_FILE)
        df_models = pd.read_csv(MODELS_FILE)
    except FileNotFoundError:
        print("Erreur : Fichiers CSV introuvables.")
        sys.exit(1)

    df_encoded = df_human.apply(encode_syllogism, axis=1)
    
    # Entropie
    df_entropy = df_encoded.groupby('task')['response_enc'].apply(
        lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True)))
    ).reset_index(name='entropy')

    # Précision (Accuracy)
    df_encoded['is_correct'] = df_encoded.apply(
        lambda row: 1 if row['response_enc'] == VALID_CONCLUSIONS.get(row['task'], 'NVC') else 0, axis=1
    )
    df_accuracy = df_encoded.groupby('task')['is_correct'].mean().reset_index(name='accuracy')
    df_accuracy['accuracy'] = df_accuracy['accuracy'] * 100

    # Modèles
    df_models_avg = df_models.groupby('task')['number_models_generated'].mean().reset_index()
    
    # Fusion
    df_merged = pd.merge(df_models_avg, df_entropy, on='task')
    df_merged = pd.merge(df_merged, df_accuracy, on='task')

    print("2. Application du K-Means (k=2)...")
    # Création de la matrice X pour l'algorithme
    X = df_merged[['entropy', 'number_models_generated', 'accuracy']]
    
    # NORMALISATION : Cruciale pour le K-Means en 3D avec des échelles différentes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement du K-Means
    kmeans = KMeans(n_clusters=2, random_state=84, n_init=10)
    df_merged['Cluster'] = kmeans.fit_predict(X_scaled)

    # Noms explicites pour les clusters basés sur l'analyse des centres de gravité
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    print("\n--- ANALYSE DES 3 FAMILLES COGNITIVES ---")
    for i, center in enumerate(cluster_centers):
        print(f"Cluster {i} -> Entropie: {center[0]:.2f} bits | Modèles: {center[1]:.2f} | Succès: {center[2]:.1f}%")
        print(f"   (Syllogismes : {', '.join(df_merged[df_merged['Cluster'] == i]['task'].tolist()[:5])}...)\n")

    print("3. Génération de la vue 3D Clusterisée...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Palette de couleurs distinctes pour les 2 clusters
    colors = ['#10b981', '#ef4444']
    
    for cluster_id in range(2):
        cluster_data = df_merged[df_merged['Cluster'] == cluster_id]
        ax.scatter(
            cluster_data['entropy'], 
            cluster_data['number_models_generated'], 
            cluster_data['accuracy'], 
            color=colors[cluster_id], 
            label=f'Cluster {cluster_id}',
            s=100, 
            edgecolors='w', 
            alpha=0.9
        )
        
    ax.set_title('Clustering K-Means (k=2)', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Entropie (Incertitude Humaine)')
    ax.set_ylabel('Modèles (Effort mReasoner)')
    ax.set_zlabel('Précision (% Succès Humain)')
    
    # Ajustement de l'angle de vue pour mieux séparer les clusters visuellement
    ax.view_init(elev=20, azim=135)
    
    plt.legend(title="Familles K-Means", loc='upper left')
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
    )
        
    print(
        "Graphique 3D sauvegardé dans :",
        OUTPUT_FILE,
    )
    plt.show()

if __name__ == '__main__':
    main()