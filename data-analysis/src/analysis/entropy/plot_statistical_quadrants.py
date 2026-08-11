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

OUTPUT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "statistical_quadrants.png",
)


os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


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

def plot_quadrant(df_merged, ax, thresh_e, thresh_m, title):
    """ Fonction générique pour tracer un quadrant basé sur des seuils précis """
    
    # 1. Catégorisation dynamique basée sur les seuils fournis
    def get_quadrant(row):
        e, m = row['entropy'], row['number_models_generated']
        if e <= thresh_e and m <= thresh_m:
            return '1. Évidence (Faible Entr. / Faible Mod.)'
        elif e <= thresh_e and m > thresh_m:
            return '2. (Faible Entr. / Fort Mod.)'
        elif e > thresh_e and m > thresh_m:
            return '3. Complexité (Forte Entr. / Fort Mod.)'
        else:
            return '4. Biais fort (Forte Entr. / Faible Mod.)'
            
    df_merged['Quadrant'] = df_merged.apply(get_quadrant, axis=1)

    palette = {
        '1. Évidence (Faible Entr. / Faible Mod.)': '#10b981',    # Vert
        '2. (Faible Entr. / Fort Mod.)': '#f59e0b',        # Orange
        '3. Complexité (Forte Entr. / Fort Mod.)': '#ef4444',    # Rouge
        '4. Biais fort (Forte Entr. / Faible Mod.)': '#8b5cf6'   # Violet
    }

    # 2. Nuage de points
    sns.scatterplot(
        data=df_merged, x='entropy', y='number_models_generated', 
        hue='Quadrant', palette=palette, alpha=0.9, s=80, edgecolor='white', ax=ax
    )

    # 3. Lignes de séparation statistiques
    ax.axvline(x=thresh_e, color='#334155', linestyle='--', linewidth=2, zorder=0)
    ax.axhline(y=thresh_m, color='#334155', linestyle='--', linewidth=2, zorder=0)

    # 4. Mathématiques : Trouver les points les plus extrêmes de chaque quadrant
    # On calcule la distance euclidienne normalisée par rapport au centre de la croix
    std_e = df_merged['entropy'].std()
    std_m = df_merged['number_models_generated'].std()
    
    df_merged['dist_to_center'] = np.sqrt(
        ((df_merged['entropy'] - thresh_e) / std_e)**2 + 
        ((df_merged['number_models_generated'] - thresh_m) / std_m)**2
    )

    # Annotation des 2 syllogismes les plus éloignés (représentatifs) de chaque quadrant
    for q in df_merged['Quadrant'].unique():
        top_points = df_merged[df_merged['Quadrant'] == q].nlargest(2, 'dist_to_center')
        for _, row in top_points.iterrows():
            ax.text(row['entropy'] + 0.03, row['number_models_generated'] + 0.03, 
                     row['task'], fontsize=9, color='black', weight='bold')

    # 5. Esthétique
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Entropie des Réponses Humaines', fontsize=12)
    ax.set_ylabel('Nombre moyen de modèles (mReasoner)', fontsize=12)
    ax.set_xlim(1.4, df_merged['entropy'].max() + 0.15)
    ax.set_ylim(1.8, df_merged['number_models_generated'].max() + 0.3)
    ax.legend(title='Analyse', loc='upper left', fontsize=9, title_fontsize=10)

def main():
    print("1. Traitement des données...")
    try:
        df_human = pd.read_csv(HUMAN_DATA_FILE)
        df_models = pd.read_csv(MODELS_FILE)
    except FileNotFoundError:
        print("Erreur : Fichiers CSV introuvables. Vérifiez les chemins.")
        sys.exit(1)

    df_encoded = df_human.apply(encode_syllogism, axis=1)
    df_entropy = df_encoded.groupby('task')['response_enc'].apply(
        lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True)))
    ).reset_index(name='entropy')

    df_models_avg = df_models.groupby('task')['number_models_generated'].mean().reset_index()
    df_merged = pd.merge(df_models_avg, df_entropy, on='task')

    # --- CALCULS STATISTIQUES RIGOUREUX ---
    mean_e = df_merged['entropy'].mean()
    mean_m = df_merged['number_models_generated'].mean()
    
    median_e = df_merged['entropy'].median()
    median_m = df_merged['number_models_generated'].median()

    print(f"Moyennes : Entropie = {mean_e:.2f}, Modèles = {mean_m:.2f}")
    print(f"Médianes : Entropie = {median_e:.2f}, Modèles = {median_m:.2f}")

    print("2. Génération de la figure comparative...")
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    sns.set_theme(style="whitegrid")

    # Trace le graphique avec séparation par la MOYENNE
    plot_quadrant(df_merged, axs[0], mean_e, mean_m, 
                  f"Séparation par la Moyenne\n(Croix à Entropie={mean_e:.2f}, Modèles={mean_m:.2f})")

    # Trace le graphique avec séparation par la MÉDIANE
    plot_quadrant(df_merged, axs[1], median_e, median_m, 
                  f"Séparation par la Médiane\n(Croix à Entropie={median_e:.2f}, Modèles={median_m:.2f})")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
    )

    print(
        "\n Matrice sauvegardée dans :",
        OUTPUT_FILE,
    )
    plt.show()

if __name__ == '__main__':
    main()