import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
from pathlib import Path

# ======================================================================
# CONFIGURATION (minimal, juste chemins)
# ======================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# src/analysis/consensus -> racine du repo
REPO_ROOT = SCRIPT_DIR.parents[2]

HUMAN_DATA_FILE = (
    REPO_ROOT
    / "data"
    / "raw"
    / "Ragni2016.csv"
)

MODEL_COUNT_FILE = (
    REPO_ROOT
    / "results"
    / "tables"
    / "mental_models"
    / "mental_models_count.csv"
)

OUTPUT_DIRECTORY = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "consensus"
)

QUADRANTS_FILE = OUTPUT_DIRECTORY / "quadrants_plot.png"

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def encode_syllogism(row):
    """ Mini-parseur autonome pour l'encodage universel """
    task_str = row['task']
    resp_str = row['response']
    
    prem1, prem2 = task_str.split('/')
    q1, t1_1, t1_2 = prem1.split(';')
    q2, t2_1, t2_2 = prem2.split(';')
    
    quant_map = {'All': 'A', 'Some': 'I', 'No': 'E', 'Some not': 'O'}
    quant1 = quant_map[q1]
    quant2 = quant_map[q2]
    
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
        df_models = pd.read_csv(MODEL_COUNT_FILE)
    except FileNotFoundError:
        print("Erreur : Fichiers CSV introuvables.")
        sys.exit(1)

    # Encodage et calcul du consensus
    df_encoded = df_human.apply(encode_syllogism, axis=1)
    df_consensus = df_encoded.groupby('task')['response_enc'].apply(
        lambda x: x.value_counts().max() / len(x)
    ).reset_index(name='consensus_rate')

    # Moyenne des modèles et fusion
    df_models_avg = df_models.groupby('task')['number_models_generated'].mean().reset_index()
    df_merged = pd.merge(df_models_avg, df_consensus, on='task')

    # --- NOUVEAU : Catégorisation des Syllogismes ---
    # On ajoute une colonne pour définir le "Quadrant Cognitif" de chaque syllogisme
    def get_category(row):
        if row['consensus_rate'] >= 0.5:
            return 'Majorité (Système 1 dominant)'
        else:
            return 'Ambiguïté (Système 2 / Dispersion)'
            
    df_merged['Category'] = df_merged.apply(get_category, axis=1)

    print("2. Génération de l'analyse en Quadrants...")
    plt.figure(figsize=(11, 8))
    sns.set_theme(style="whitegrid")

    # Nuage de points coloré selon la catégorie
    ax = sns.scatterplot(
        data=df_merged, 
        x='consensus_rate', 
        y='number_models_generated', 
        hue='Category',
        palette={'Majorité (Système 1 dominant)': '#10b981', 'Ambiguïté (Système 2 / Dispersion)': '#f43f5e'}, # Vert et Rose
        alpha=0.8, 
        s=90, 
        edgecolor='white',
        linewidth=1
    )

    # --- La Fameuse Ligne de Séparation à 0.5 ---
    plt.axvline(x=0.5, color='#64748b', linestyle='--', linewidth=2, zorder=0)
    
    # On ajoute un fond coloré léger pour bien séparer les zones visuellement
    plt.axvspan(0, 0.5, facecolor='#fef2f2', alpha=0.5, zorder=-1) # Fond rosé à gauche
    plt.axvspan(0.5, 1.05, facecolor='#ecfdf5', alpha=0.5, zorder=-1) # Fond verdâtre à droite


    # Esthétique
    plt.title('Frontière : Ambiguïté vs Majorité\n(Séparation à 50% de Consensus)', 
            fontsize=15, pad=20, fontweight='bold')
    plt.xlabel('Taux de Consensus Humain', fontsize=12)
    plt.ylabel('Nombre moyen de modèles générés (mReasoner)', fontsize=12)
    
    plt.xlim(0, 1.05)
    
    # Ajustement de la légende
    plt.legend(title='Profil du Syllogisme', loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(
        str(QUADRANTS_FILE),
        dpi=300,
    )
    print(f"\n✅ Graphique sauvegardé dans : {QUADRANTS_FILE}")
    plt.show()

if __name__ == '__main__':
    main()