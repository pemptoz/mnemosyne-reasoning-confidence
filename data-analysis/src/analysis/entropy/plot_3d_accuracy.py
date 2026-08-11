import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

ANALYSIS_3D_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "analyse_3d.png",
)

ANALYSIS_2D_OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "analyse_2d_croisee.png",
)


os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


# Dictionnaire des réponses logiquement valides (Logique standard)
# Les syllogismes n'étant pas dans cette liste ont pour réponse correcte "NVC" (No Valid Conclusion)
VALID_CONCLUSIONS = {
    'AA1': 'Aac', 'EA1': 'Eac', 'AE2': 'Eac', 'EA2': 'Eac',
    'AA3': 'Iac', 'IA3': 'Iac', 'AI3': 'Iac', 'EA3': 'Oac',
    'OA3': 'Oac', 'EI3': 'Oac', 'AA4': 'Ica', 'AE4': 'Eac',
    'IA4': 'Ica', 'EA4': 'Oac', 'EI4': 'Oac'
}

def encode_syllogism(row):
    """ Parseur autonome pour l'encodage universel """
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
        print("Erreur : Fichiers CSV introuvables. Vérifiez les chemins.")
        sys.exit(1)

    # Encodage des tâches[cite: 2]
    df_encoded = df_human.apply(encode_syllogism, axis=1)
    
    # 2. Calcul de l'Entropie
    df_entropy = df_encoded.groupby('task')['response_enc'].apply(
        lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True)))
    ).reset_index(name='entropy')

    # 3. Calcul du Pourcentage de Bonnes Réponses (Accuracy)
    df_encoded['is_correct'] = df_encoded.apply(
        lambda row: 1 if row['response_enc'] == VALID_CONCLUSIONS.get(row['task'], 'NVC') else 0, 
        axis=1
    )
    df_accuracy = df_encoded.groupby('task')['is_correct'].mean().reset_index(name='accuracy')
    # On convertit en pourcentage (0 à 100)
    df_accuracy['accuracy'] = df_accuracy['accuracy'] * 100

    # 4. Calcul de la moyenne des Modèles mReasoner
    df_models_avg = df_models.groupby('task')['number_models_generated'].mean().reset_index()
    
    # 5. Fusion de toutes les données
    df_merged = pd.merge(df_models_avg, df_entropy, on='task')
    df_merged = pd.merge(df_merged, df_accuracy, on='task')

    print("2. Génération de la vue 3D...")
    
    # --- FIGURE 1 : GRAPH 3D ---
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    # Coloration basée sur l'accuracy (rouge = faible, vert = élevé)
    scatter = ax3d.scatter(
        df_merged['entropy'], 
        df_merged['number_models_generated'], 
        df_merged['accuracy'], 
        c=df_merged['accuracy'], 
        cmap='RdYlGn', 
        s=80, 
        edgecolors='w', 
        alpha=0.9
    )
    
    ax3d.set_title('Espace Cognitif 3D : Modèles vs Entropie vs Bonnes Réponses', fontsize=14, fontweight='bold', pad=20)
    ax3d.set_xlabel('Entropie Humaine (Bits)')
    ax3d.set_ylabel('Modèles mReasoner')
    ax3d.set_zlabel('% Bonnes Réponses')
    
    cbar = plt.colorbar(scatter, ax=ax3d, pad=0.1)
    cbar.set_label('% de Bonnes Réponses (Précision Humaine)')
    
    plt.savefig(
        ANALYSIS_3D_OUTPUT_FILE,
        dpi=300,
    )

    print("3. Génération des vues 2D (Projections)...")
    
    # --- FIGURE 2 : GRAPHIQUES 2D (1x3) ---
    fig2d, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig2d.suptitle("Projections 2D : Corrélations Croisées", fontsize=16, fontweight='bold')
    sns.set_theme(style="whitegrid")

    # Graphe 1 : Entropie vs Modèles (Celui qu'on connait)
    sns.regplot(data=df_merged, x='entropy', y='number_models_generated', 
                ax=axs[0], scatter_kws={'color':'#6366f1', 'alpha':0.7}, line_kws={'color':'#f43f5e'})
    axs[0].set_title('Entropie vs Modèles mReasoner')
    axs[0].set_xlabel('Entropie (Bits)')
    axs[0].set_ylabel('Nombre de Modèles')

    # Graphe 2 : Entropie vs Accuracy
    sns.regplot(data=df_merged, x='entropy', y='accuracy', 
                ax=axs[1], scatter_kws={'color':'#10b981', 'alpha':0.7}, line_kws={'color':'#f43f5e'})
    axs[1].set_title('Entropie vs Précision Humaine')
    axs[1].set_xlabel('Entropie (Bits)')
    axs[1].set_ylabel('% Bonnes Réponses')

    # Graphe 3 : Modèles vs Accuracy
    sns.regplot(data=df_merged, x='number_models_generated', y='accuracy', 
                ax=axs[2], scatter_kws={'color':'#f59e0b', 'alpha':0.7}, line_kws={'color':'#f43f5e'})
    axs[2].set_title('Modèles mReasoner vs Précision Humaine')
    axs[2].set_xlabel('Nombre de Modèles')
    axs[2].set_ylabel('% Bonnes Réponses')

    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(
        ANALYSIS_2D_OUTPUT_FILE,
        dpi=300,
    )
    
    print(
        "\nSuccès ! Fichiers générés :",
        ANALYSIS_3D_OUTPUT_FILE,
        "et",
        ANALYSIS_2D_OUTPUT_FILE,
    )
    plt.show()

if __name__ == '__main__':
    main()