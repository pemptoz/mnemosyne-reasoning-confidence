import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
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
    "advanced_entropy_correlations.png",
)


os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


def exp_func(x, a, b, c):
    # CORRECTION MATHÉMATIQUE : On décale l'axe X pour que l'exponentielle 
    # commence son point d'origine exactement à l'entropie minimale (1.5)
    return a * np.exp(b * (x - 1.5)) + c

def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

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
    print("1. Traitement des données et calcul de l'entropie...")
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

    x = df_merged['entropy'].values
    y = df_merged['number_models_generated'].values
    
    # Plage de X strictement restreinte (1.5 à Max)
    xp = np.linspace(1.5, max(x) + 0.1, 100)

    print("2. Génération de la grille d'analyse multi-modèles...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploration Avancée : Entropie Cognitive vs Modèles mReasoner", fontsize=16, fontweight='bold', y=0.98)
    sns.set_theme(style="whitegrid")

    # PLOT 1 : Données brutes
    axs[0, 0].scatter(x, y, color='#8b5cf6', alpha=0.8, edgecolor='white', s=60)
    axs[0, 0].set_title('1. Données Brutes (Aucun lissage)', fontsize=12)

    # PLOT 2 : Linéaire (Strictement Degré 1)
    axs[0, 1].scatter(x, y, color='#8b5cf6', alpha=0.6, edgecolor='white', s=60)
    coefs_lin = np.polyfit(x, y, 1)
    a, b = coefs_lin
    p1 = np.poly1d(coefs_lin)

    axs[0, 1].plot(xp, p1(xp), color='#10b981', linewidth=2.5)
    axs[0, 1].set_title(
        f'2. Linéaire : y = {a:.3f}x + {b:.3f}',
        fontsize=11
    )

    # PLOT 3 : Quadratique (Strictement Degré 2)
    axs[1, 0].scatter(x, y, color='#8b5cf6', alpha=0.6, edgecolor='white', s=60)
    coefs_quad = np.polyfit(x, y, 2)
    a, b, c = coefs_quad
    p2 = np.poly1d(coefs_quad)

    axs[1, 0].plot(xp, p2(xp), color='#f43f5e', linewidth=2.5)
    axs[1, 0].set_title(
        f'3. Quadratique : y = {a:.3f}x² + {b:.3f}x + {c:.3f}',
        fontsize=10
    )

    # PLOT 4 : Ajustement exponentiel
    axs[1, 1].scatter(x, y, color='#8b5cf6', alpha=0.6,
                    edgecolor='white', s=60)

    try:
        popt, _ = curve_fit(
            exp_func,
            x,
            y,
            p0=(0.5, 1.0, 2.0),
            maxfev=5000
        )

        a, b, c = popt

        axs[1, 1].plot(
            xp,
            exp_func(xp, *popt),
            color='#f59e0b',
            linewidth=2.5
        )

        axs[1, 1].set_title(
            f'4. Exp. : y = {a:.3f}·e^({b:.3f}(x-1.5)) + {c:.3f}',
            fontsize=10
        )

    except Exception as e:
        axs[1, 1].set_title("4. Exponentiel (échec de convergence)", fontsize=12)
        print(f"Erreur exponentielle : {e}")

    for ax in axs.flat:
        ax.set_xlim(1.4, max(x) + 0.1)
        ax.set_ylim(1.8, max(y) + 0.5)
        ax.set_xlabel('Entropie (Bits)')
        ax.set_ylabel('Nombre de Modèles (Moyenne)')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
    )
    print(
        "\nGraphiques sauvegardés dans :",
        OUTPUT_FILE,
    )
    plt.show()

if __name__ == '__main__':
    main()