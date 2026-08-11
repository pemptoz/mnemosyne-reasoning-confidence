import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from pathlib import Path

# ======================================================================
# CONFIGURATION (minimal, juste chemins)
# ======================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# src/analysis/consensus -> racine du repo
REPO_ROOT = SCRIPT_DIR.parents[4]

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

ADVANCED_CORRELATIONS_FILE = OUTPUT_DIRECTORY / "advanced_correlations.png"

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


def exp_func(x, a, b, c):
    # Modèle exponentiel standard
    return a * np.exp(b * x) + c

def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def encode_syllogism(row):
    """ Parseur autonome pour l'encodage universel """
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
        print("Erreur : Fichiers CSV introuvables. Vérifiez les chemins.")
        sys.exit(1)

    df_encoded = df_human.apply(encode_syllogism, axis=1)
    
    # Calcul du taux de consensus
    df_consensus = df_encoded.groupby('task')['response_enc'].apply(
        lambda x: x.value_counts().max() / len(x)
    ).reset_index(name='consensus_rate')

    df_models_avg = df_models.groupby('task')['number_models_generated'].mean().reset_index()
    df_merged = pd.merge(df_models_avg, df_consensus, on='task')

    x = df_merged['consensus_rate'].values
    y = df_merged['number_models_generated'].values
    xp = np.linspace(min(x) - 0.05, max(x) + 0.05, 100) # Points pour tracer les courbes lisses

    print("2. Génération de la grille d'analyse multi-modèles...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Exploration Avancée : Concentration Humaine vs Modèles mReasoner', fontsize=16, fontweight='bold', y=0.98)
    sns.set_theme(style="whitegrid")

    # --- PLOT 1 : Données Brutes ---
    axs[0, 0].scatter(x, y, color='#64748b', alpha=0.8, edgecolor='white', s=60)
    axs[0, 0].set_title('1. Données Brutes (Aucun lissage)', fontsize=13)

    # --- PLOT 2 : Ajustement Linéaire (y = ax + b) ---
    axs[0, 1].scatter(x, y, color='#6366f1', alpha=0.6, edgecolor='white', s=60)
    coefs_lin = np.polyfit(x, y, 1)
    a_l, b_l = coefs_lin
    p1 = np.poly1d(coefs_lin)
    axs[0, 1].plot(xp, p1(xp), color='#10b981', linewidth=2.5)
    
    sign_b_l = "+" if b_l >= 0 else "-"
    axs[0, 1].set_title(f"2. Linéaire : y = {a_l:.2f}x {sign_b_l} {abs(b_l):.2f}", fontsize=13)

    # --- PLOT 3 : Ajustement Quadratique (y = ax² + bx + c) ---
    axs[1, 0].scatter(x, y, color='#6366f1', alpha=0.6, edgecolor='white', s=60)
    coefs_quad = np.polyfit(x, y, 2)
    a_q, b_q, c_q = coefs_quad
    p2 = np.poly1d(coefs_quad)
    axs[1, 0].plot(xp, p2(xp), color='#f43f5e', linewidth=2.5)
    
    sign_b_q = "+" if b_q >= 0 else "-"
    sign_c_q = "+" if c_q >= 0 else "-"
    axs[1, 0].set_title(f"3. Quadratique : y = {a_q:.2f}x² {sign_b_q} {abs(b_q):.2f}x {sign_c_q} {abs(c_q):.2f}", fontsize=13)

    # --- PLOT 4 : Ajustement Exponentiel (y = a*exp(bx) + c) ---
    axs[1, 1].scatter(x, y, color='#6366f1', alpha=0.6, edgecolor='white', s=60)
    try:
        # Estimations initiales adaptées pour une courbe décroissante
        popt, _ = curve_fit(exp_func, x, y, p0=(5.0, -2.0, 2.0), maxfev=5000)
        a_e, b_e, c_e = popt
        axs[1, 1].plot(xp, exp_func(xp, *popt), color='#f59e0b', linewidth=2.5)
        
        sign_c_e = "+" if c_e >= 0 else "-"
        axs[1, 1].set_title(f"4. Exponentiel : y = {a_e:.2f} * exp({b_e:.2f}x) {sign_c_e} {abs(c_e):.2f}", fontsize=13)
    except Exception as e:
        axs[1, 1].set_title('4. Exponentiel (Échec de convergence)', fontsize=13)
        print(f"Erreur exponentielle : {e}")

    for ax in axs.flat:
        ax.set_xlim(0, 1.05)
        ax.set_ylim(1.5, 5.0)
        ax.set_xlabel('Taux de Consensus Humain', fontsize=11)
        ax.set_ylabel('Nombre de Modèles (Moyenne)', fontsize=11)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(
        str(ADVANCED_CORRELATIONS_FILE),
        dpi=300,
    )
    print(f"\n✅ Graphique sauvegardé dans : {ADVANCED_CORRELATIONS_FILE}")
    plt.show()

if __name__ == '__main__':
    main()