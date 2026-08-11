import json
import os
import sys
from pathlib import Path

import ccobra
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mreasoner


PARAMETER_LOG_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "log_full.json"
)

CCL_EXECUTABLE = (
    PROJECT_ROOT
    / ".ccl"
    / "ccl"
    / "lx86cl64"
)

MREASONER_SOURCE_ROOT = (
    PROJECT_ROOT
    / ".mreasoner"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "mental_models"
    / "mental_models_count.csv"
)

DEBUG_TRACE_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "debug_first_trace.txt"
)


def syllog_to_premises(syllog):
    """ Convertit un encodage (ex: AA1) en phrases textuelles pour le LISP. """
    template_quant = {'A': 'All {} are {}', 'I': 'Some {} are {}', 'E': 'No {} are {}', 'O': 'Some {} are not {}'}
    template_fig = {'1': [['A', 'B'], ['B', 'C']], '2': [['B', 'A'], ['C', 'B']], '3': [['A', 'B'], ['C', 'B']], '4': [['B', 'A'], ['B', 'C']]}
    prem1 = template_quant[syllog[0]].format(*template_fig[syllog[-1]][0])
    prem2 = template_quant[syllog[1]].format(*template_fig[syllog[-1]][1])
    return [prem1, prem2]

def parse_models_count(trace_str):
    """
    Parse la trace LISP pour compter le nombre de modèles.
    Au lieu d'un simple `.count("MODEL")` bruité, nous ciblons les tags d'actions
    spécifiques du traceur mReasoner (ex: la construction ou l'ajout d'un modèle).
    """
    if not trace_str:
        return 0
        
    # On compte le nombre d'occurrences de l'objet LISP "#<Q-MODEL" 
    # qui représente un modèle mental dans la mémoire de LISP.
    count = trace_str.count("#<Q-MODEL")
    
    return max(1, count)

def main():
    # 1. Chargement des meilleurs paramètres identifiés par CCOBRA
    print("Chargement du log JSON des paramètres (log_full.json)...")
    try:
        with open(
            PARAMETER_LOG_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            log_data = json.load(f)['mReasoner'] 
    except FileNotFoundError:
        print("Erreur : Fichier 'log_full.json' introuvable.")
        return

    # Extraction sécurisée : associe chaque Subject ID à son meilleur paramètre
    subject_params = {}
    for subj_id, log in log_data.items():
        params_list = log.get('best_params', [mreasoner.DEFAULT_PARAMS])
        subject_params[str(subj_id)] = params_list[0]

    # 2. Initialisation du moteur LISP
    print("Démarrage du processus ClozureCL mReasoner...")
    mreas_path = mreasoner.source_path(
        str(MREASONER_SOURCE_ROOT)
    )

    mr = mreasoner.MReasoner(
        str(CCL_EXECUTABLE),
        mreas_path,
    )


    # 3. Préparation du Cache RAM (Mémoïsation)
    memo_cache = {}
    n_samples = 3 # Lissage stochastique
    results = []
    first_trace_saved = False

    print(f"Nombre de participants à analyser : {len(subject_params)}")

    # 4. Boucle principale (BYPASS COMPLET DU DATASET CCOBRA !)
    print("Début de l'analyse Post-Hoc...")
    for subj_id, current_params in tqdm(subject_params.items(), desc="Analyse des participants"):
        
        param_tuple = (
            current_params.get('epsilon', 0.0), 
            current_params.get('lambda', 4.0), 
            current_params.get('omega', 1.0), 
            current_params.get('sigma', 0.0)
        )

        # On itère directement sur les 64 syllogismes officiels de la bibliothèque
        for task_enc in ccobra.syllogistic.SYLLOGISMS:
            
            cache_key = (task_enc, *param_tuple)
            
            if cache_key not in memo_cache:
                premises = syllog_to_premises(task_enc)
                sample_counts = []
                
                for _ in range(n_samples):
                    trace_str = mr.query_trace(premises, current_params)
                    
                    # Dump de la toute première trace pour vérifier manuellement les mots-clés
                    if not first_trace_saved and trace_str:
                        DEBUG_TRACE_FILE.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        with open(
                            DEBUG_TRACE_FILE,
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(trace_str)
                        first_trace_saved = True
                        
                    count = parse_models_count(trace_str)
                    sample_counts.append(count)
                    
                memo_cache[cache_key] = np.mean(sample_counts)
            
            results.append({
                'subject_id': subj_id,
                'task': task_enc,
                'number_models_generated': round(memo_cache[cache_key], 2)
            })

    # 5. Fermeture et Sauvegarde
    mr.terminate()
    df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )
    
    print("\n Succès ! Données générées dans 'mental_models_count.csv'.")
    print("Consulter 'debug_first_trace.txt' pour vérifier que le parser LISP compte les bons mots-clés.")

if __name__ == '__main__':
    main()