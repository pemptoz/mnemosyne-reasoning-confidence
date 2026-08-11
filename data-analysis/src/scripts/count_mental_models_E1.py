import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


# ======================================================================
# Chemins
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mreasoner


CCL_EXECUTABLE = str(
    PROJECT_ROOT
    / ".ccl"
    / "ccl"
    / "lx86cl64"
)

MREASONER_SOURCE_ROOT = str(
    PROJECT_ROOT
    / ".mreasoner"
)

PARAMETER_LOG_FILE = str(
    PROJECT_ROOT
    / "results"
    / "logs"
    / "full_log_E1.json"
)

PREDICTION_CACHE_FILE = str(
    PROJECT_ROOT
    / "data"
    / "cache"
    / "pymreasoner_2"
    / "mreasoner_conditional_cache_E1.npz"
)


# ======================================================================
# Fichiers produits
# ======================================================================

DETAILED_OUTPUT_FILE = str(
    PROJECT_ROOT
    / "results"
    / "tables"
    / "mental_models"
    / "mental_models_count_E1.csv"
)

SUBJECT_SUMMARY_FILE = str(
    PROJECT_ROOT
    / "results"
    / "tables"
    / "mental_models"
    / "mental_models_mean_by_subject_E1.csv"
)

POSTHOC_CACHE_FILE = str(
    PROJECT_ROOT
    / "data"
    / "cache"
    / "mental_models_posthoc_cache_E1.json"
)




# ======================================================================
# Configuration du calcul
# ======================================================================

# Nombre d'appels stochastiques de mReasoner pour chaque :
#
#     configuration de paramètres × tâche
#
# Commence avec 3 pour tester.
# Tu pourras ensuite utiliser 10 ou 20 pour une moyenne plus stable.
N_SAMPLES = 3

# "selected" :
#     utilise epsilon, lambda, omega et sigma stockés directement dans
#     l'entrée du participant.
#
# "all_best" :
#     utilise toutes les configurations contenues dans best_params,
#     puis moyenne le nombre de modèles sur toutes ces configurations.
PARAMETER_SELECTION = "selected"

# Afficher quelques traces complètes afin de vérifier le comptage.
DEBUG_TRACES = True

# Nombre maximal de traces affichées.
MAX_DEBUG_TRACES = 3

# Sauvegarde du cache toutes les N nouvelles combinaisons calculées.
SAVE_CACHE_EVERY = 20


# ======================================================================
# Sauvegarde JSON atomique
# ======================================================================

def atomic_json_dump(data, output_path):
    """
    Sauvegarde un fichier JSON de manière atomique.
    """
    output_directory = (
        os.path.dirname(output_path)
        or "."
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".temporary-model-count-",
        suffix=".json",
        dir=output_directory,
        text=True,
    )

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                data,
                output_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            output_file.flush()
            os.fsync(
                output_file.fileno()
            )

        os.replace(
            temporary_path,
            output_path,
        )

        temporary_path = None

    finally:
        if (
            temporary_path is not None
            and os.path.exists(temporary_path)
        ):
            os.remove(temporary_path)


# ======================================================================
# Chargement des paramètres
# ======================================================================

def normalize_params(params):
    """
    Extrait et valide epsilon, lambda, omega et sigma.
    """
    parameter_names = (
        "epsilon",
        "lambda",
        "omega",
        "sigma",
    )

    missing_parameters = [
        parameter_name
        for parameter_name in parameter_names
        if parameter_name not in params
    ]

    if missing_parameters:
        raise ValueError(
            "Paramètres absents du journal : "
            f"{missing_parameters}. Entrée reçue : {params!r}"
        )

    return {
        parameter_name: float(
            params[parameter_name]
        )
        for parameter_name in parameter_names
    }


def params_to_tuple(params):
    """
    Produit une représentation stable des paramètres.

    L'arrondi évite de distinguer inutilement :
        0.3
        0.30000000000000004
    """
    return (
        round(float(params["epsilon"]), 12),
        round(float(params["lambda"]), 12),
        round(float(params["omega"]), 12),
        round(float(params["sigma"]), 12),
    )


def load_subject_logs():
    """
    Charge la section mReasoner de full_log_E1.json.
    """
    if not os.path.isfile(PARAMETER_LOG_FILE):
        raise FileNotFoundError(
            "Fichier de paramètres introuvable : "
            f"{PARAMETER_LOG_FILE}"
        )

    with open(
        PARAMETER_LOG_FILE,
        "r",
        encoding="utf-8",
    ) as input_file:
        full_log = json.load(
            input_file
        )

    if not isinstance(full_log, dict):
        raise ValueError(
            "Le contenu de full_log_E1.json "
            "doit être un objet JSON."
        )

    if "mReasoner" not in full_log:
        raise KeyError(
            "La clé 'mReasoner' est absente de "
            f"{PARAMETER_LOG_FILE}."
        )

    subject_logs = full_log[
        "mReasoner"
    ]

    if not isinstance(subject_logs, dict):
        raise ValueError(
            "La section 'mReasoner' doit être "
            "un dictionnaire."
        )

    return {
        str(subject_id): subject_log
        for subject_id, subject_log
        in subject_logs.items()
    }


def extract_parameter_sets(subject_log):
    """
    Retourne les paramètres à utiliser pour un participant.

    En mode selected, une seule configuration est utilisée.

    En mode all_best, toutes les configurations ex aequo de
    best_params sont utilisées.
    """
    selected_params = normalize_params(
        subject_log
    )

    if PARAMETER_SELECTION == "selected":
        return [selected_params]

    if PARAMETER_SELECTION != "all_best":
        raise ValueError(
            "PARAMETER_SELECTION doit être "
            "'selected' ou 'all_best'."
        )

    raw_best_params = subject_log.get(
        "best_params",
        [],
    )

    if not raw_best_params:
        return [selected_params]

    parameter_sets = []
    already_seen = set()

    for raw_params in raw_best_params:
        params = normalize_params(
            raw_params
        )

        key = params_to_tuple(
            params
        )

        if key in already_seen:
            continue

        already_seen.add(key)
        parameter_sets.append(params)

    if not parameter_sets:
        return [selected_params]

    return parameter_sets


# ======================================================================
# Chargement des quatre tâches uniques
# ======================================================================

def load_unique_tasks_from_prediction_cache():
    """
    Charge les tâches réellement utilisées pendant le benchmark depuis
    mreasoner_conditional_cache_E1.npz.

    Le cache contient les prémisses dans premises_json.

    Returns
    -------
    list(dict)


$$
{
                "task_id": 1,
                "premises": (
                    "All B are C",
                    "No A are C"
                )
            },
            ...
$$


    """
    if not os.path.isfile(PREDICTION_CACHE_FILE):
        raise FileNotFoundError(
            "Cache des tâches introuvable : "
            f"{PREDICTION_CACHE_FILE}"
        )

    with np.load(
        PREDICTION_CACHE_FILE,
        allow_pickle=False,
    ) as cache_data:
        if "premises_json" not in cache_data.files:
            raise KeyError(
                "La clé 'premises_json' est absente de "
                f"{PREDICTION_CACHE_FILE}."
            )

        serialized_premises = (
            cache_data["premises_json"]
            .astype(str)
            .tolist()
        )

    unique_tasks = []
    already_seen = set()

    for serialized_task in serialized_premises:
        premises = json.loads(
            serialized_task
        )

        if not isinstance(
            premises,
            (list, tuple),
        ):
            raise ValueError(
                "Format de tâche invalide dans le cache : "
                f"{premises!r}"
            )

        premises = tuple(
            str(premise).strip()
            for premise in premises
        )

        if len(premises) != 2:
            raise ValueError(
                "Chaque tâche doit contenir exactement "
                f"deux prémisses : {premises!r}"
            )

        if premises in already_seen:
            continue

        already_seen.add(premises)

        unique_tasks.append({
            "task_id": len(unique_tasks) + 1,
            "premises": premises,
        })

    if not unique_tasks:
        raise RuntimeError(
            "Aucune tâche n'a été trouvée dans le cache."
        )

    return unique_tasks


# ======================================================================
# Comptage des modèles dans la trace
# ======================================================================

def parse_models_count(trace_string):
    """
    Compte le nombre d'occurrences de #<Q-MODEL dans la trace Lisp.

    Contrairement à l'ancien code, cette fonction ne force pas le
    résultat à être au moins égal à 1.

    Ainsi :
        trace vide                    -> 0
        aucun #<Q-MODEL              -> 0
        deux occurrences #<Q-MODEL   -> 2
    """
    if trace_string is None:
        return 0

    trace_string = str(
        trace_string
    )

    if not trace_string.strip():
        return 0

    return trace_string.upper().count(
        "#<Q-MODEL"
    )


# ======================================================================
# Cache post-hoc
# ======================================================================

def load_posthoc_cache():
    """
    Charge les résultats déjà calculés.

    Le cache permet notamment de réutiliser le résultat lorsque plusieurs
    participants ont les mêmes paramètres.
    """
    if not os.path.isfile(
        POSTHOC_CACHE_FILE
    ):
        return {}

    try:
        with open(
            POSTHOC_CACHE_FILE,
            "r",
            encoding="utf-8",
        ) as input_file:
            cache = json.load(
                input_file
            )

        if not isinstance(cache, dict):
            return {}

        print(
            "Cache post-hoc chargé : "
            f"{len(cache)} entrée(s)."
        )

        return cache

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ) as error:
        print(
            "WARNING: impossible de charger le "
            f"cache post-hoc : {error!r}"
        )

        return {}


def make_posthoc_cache_key(
    premises,
    params,
):
    """
    Produit une clé stable pour une tâche et une configuration.
    """
    epsilon, lambda_, omega, sigma = (
        params_to_tuple(params)
    )

    payload = {
        "cache_version": 1,
        "count_rule":
            "number_of_Q_MODEL_occurrences",
        "premises": list(premises),
        "parameters": {
            "epsilon": epsilon,
            "lambda": lambda_,
            "omega": omega,
            "sigma": sigma,
        },
        "n_samples": int(N_SAMPLES),
    }

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ======================================================================
# Calcul du nombre de modèles
# ======================================================================

def compute_models_count(
    mr,
    premises,
    params,
    posthoc_cache,
    debug_state,
):
    """
    Appelle mReasoner N_SAMPLES fois pour une tâche et une configuration.

    Retourne :
        - la moyenne ;
        - l'écart-type ;
        - le minimum ;
        - le maximum ;
        - les résultats de chaque échantillon.
    """
    cache_key = make_posthoc_cache_key(
        premises=premises,
        params=params,
    )

    if cache_key in posthoc_cache:
        cached_result = posthoc_cache[
            cache_key
        ]

        return {
            "mean": float(
                cached_result["mean"]
            ),
            "std": float(
                cached_result["std"]
            ),
            "minimum": int(
                cached_result["minimum"]
            ),
            "maximum": int(
                cached_result["maximum"]
            ),
            "samples": [
                int(value)
                for value
                in cached_result["samples"]
            ],
            "from_cache": True,
        }

    sample_counts = []

    for sample_index in range(
        N_SAMPLES
    ):
        trace_string = mr.query_trace(
            list(premises),
            param_dict=params,
        )

        model_count = parse_models_count(
            trace_string
        )

        sample_counts.append(
            model_count
        )

        if (
            DEBUG_TRACES
            and debug_state["printed"]
            < MAX_DEBUG_TRACES
        ):
            print("\n" + "=" * 80)
            print("TRACE DE VÉRIFICATION")
            print("=" * 80)
            print(
                "Prémisses :",
                premises,
            )
            print(
                "Paramètres :",
                params,
            )
            print(
                "Échantillon :",
                sample_index + 1,
                "/",
                N_SAMPLES,
            )
            print(
                "Nombre de #<Q-MODEL détectés :",
                model_count,
            )
            print("Trace brute :")
            print(trace_string)
            print("=" * 80)

            debug_state["printed"] += 1

    sample_array = np.asarray(
        sample_counts,
        dtype=np.float64,
    )

    result = {
        "mean": float(
            np.mean(sample_array)
        ),
        "std": float(
            np.std(
                sample_array,
                ddof=0,
            )
        ),
        "minimum": int(
            np.min(sample_array)
        ),
        "maximum": int(
            np.max(sample_array)
        ),
        "samples": [
            int(value)
            for value in sample_counts
        ],
        "from_cache": False,
    }

    posthoc_cache[cache_key] = {
        "premises": list(premises),
        "parameters": params,
        "n_samples": N_SAMPLES,
        "mean": result["mean"],
        "std": result["std"],
        "minimum": result["minimum"],
        "maximum": result["maximum"],
        "samples": result["samples"],
    }

    return result


# ======================================================================
# Programme principal
# ======================================================================

def main():
    print("=" * 80)
    print("NOMBRE DE MODÈLES GÉNÉRÉS PAR PARTICIPANT")
    print("=" * 80)

    print(
        "Fichier des paramètres :",
        PARAMETER_LOG_FILE,
    )

    print(
        "Cache contenant les tâches :",
        PREDICTION_CACHE_FILE,
    )

    print(
        "Nombre de répétitions par tâche :",
        N_SAMPLES,
    )

    print(
        "Sélection des paramètres :",
        PARAMETER_SELECTION,
    )

    # ------------------------------------------------------------------
    # Vérification des fichiers et chemins
    # ------------------------------------------------------------------

    if not os.path.isfile(
        CCL_EXECUTABLE
    ):
        raise FileNotFoundError(
            "Exécutable ClozureCL introuvable : "
            f"{CCL_EXECUTABLE}"
        )

    MREASONER_DIRECTORY = mreasoner.source_path(
        MREASONER_SOURCE_ROOT
    )


    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------

    subject_logs = load_subject_logs()

    unique_tasks = (
        load_unique_tasks_from_prediction_cache()
    )

    posthoc_cache = load_posthoc_cache()

    print(
        "\nNombre de participants :",
        len(subject_logs),
    )

    print(
        "Nombre de tâches uniques :",
        len(unique_tasks),
    )

    print("\nTâches trouvées :")

    for task in unique_tasks:
        print(
            f"  Tâche {task['task_id']} : "
            f"{task['premises'][0]!r} / "
            f"{task['premises'][1]!r}"
        )

    if len(unique_tasks) != 4:
        print(
            "\nWARNING: le cache ne contient pas exactement "
            f"4 tâches, mais {len(unique_tasks)}."
        )

    # ------------------------------------------------------------------
    # Initialisation de mReasoner
    # ------------------------------------------------------------------

    print(
        "\nDémarrage du processus mReasoner..."
    )

    mr = mreasoner.MReasoner(
        CCL_EXECUTABLE,
        MREASONER_DIRECTORY,
    )

    detailed_results = []

    debug_state = {
        "printed": 0,
    }

    new_cache_entries = 0

    # ------------------------------------------------------------------
    # Boucle participants × tâches réelles
    # ------------------------------------------------------------------

    try:
        participant_iterator = tqdm(
            subject_logs.items(),
            total=len(subject_logs),
            desc="Participants",
        )

        for subject_id, subject_log in participant_iterator:
            selected_params = normalize_params(
                subject_log
            )

            parameter_sets = extract_parameter_sets(
                subject_log
            )

            for task in unique_tasks:
                task_id = task[
                    "task_id"
                ]

                premises = task[
                    "premises"
                ]

                results_for_parameter_sets = []

                for params in parameter_sets:
                    cache_key = make_posthoc_cache_key(
                        premises=premises,
                        params=params,
                    )

                    was_already_cached = (
                        cache_key in posthoc_cache
                    )

                    model_count_result = (
                        compute_models_count(
                            mr=mr,
                            premises=premises,
                            params=params,
                            posthoc_cache=posthoc_cache,
                            debug_state=debug_state,
                        )
                    )

                    results_for_parameter_sets.append(
                        model_count_result
                    )

                    if not was_already_cached:
                        new_cache_entries += 1

                        if (
                            new_cache_entries
                            % SAVE_CACHE_EVERY
                            == 0
                        ):
                            atomic_json_dump(
                                posthoc_cache,
                                POSTHOC_CACHE_FILE,
                            )

                # En mode selected, la liste contient un seul résultat.
                #
                # En mode all_best, on moyenne le résultat obtenu pour
                # toutes les configurations optimales.
                mean_models = float(
                    np.mean([
                        result["mean"]
                        for result
                        in results_for_parameter_sets
                    ])
                )

                mean_standard_deviation = float(
                    np.mean([
                        result["std"]
                        for result
                        in results_for_parameter_sets
                    ])
                )

                minimum_models = min(
                    result["minimum"]
                    for result
                    in results_for_parameter_sets
                )

                maximum_models = max(
                    result["maximum"]
                    for result
                    in results_for_parameter_sets
                )

                detailed_results.append({
                    "subject_id": str(subject_id),
                    "task": int(task_id),
                    "premise_1": premises[0],
                    "premise_2": premises[1],
                    "number_models_generated":
                        round(mean_models, 4),
                    "std_models_generated":
                        round(
                            mean_standard_deviation,
                            4,
                        ),
                    "minimum_models_generated":
                        int(minimum_models),
                    "maximum_models_generated":
                        int(maximum_models),
                    "n_samples":
                        int(N_SAMPLES),
                    "n_parameter_sets_used":
                        len(parameter_sets),
                    "epsilon":
                        selected_params["epsilon"],
                    "lambda":
                        selected_params["lambda"],
                    "omega":
                        selected_params["omega"],
                    "sigma":
                        selected_params["sigma"],
                })

        # Sauvegarde finale du cache.
        atomic_json_dump(
            posthoc_cache,
            POSTHOC_CACHE_FILE,
        )

    finally:
        print(
            "\nFermeture du processus mReasoner..."
        )

        try:
            mr.terminate()

        except Exception as error:
            print(
                "WARNING: impossible de fermer "
                f"mReasoner proprement : {error!r}"
            )

    # ------------------------------------------------------------------
    # Fichier détaillé : participant × tâche
    # ------------------------------------------------------------------

    detailed_dataframe = pd.DataFrame(
        detailed_results
    )

    detailed_dataframe = (
        detailed_dataframe.sort_values(
            by=[
                "subject_id",
                "task",
            ]
        )
    )

    detailed_dataframe.to_csv(
        DETAILED_OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Fichier résumé : moyenne par participant
    # ------------------------------------------------------------------

    subject_summary = (
        detailed_dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            mean_number_models_generated=(
                "number_models_generated",
                "mean",
            ),
            std_across_tasks=(
                "number_models_generated",
                "std",
            ),
            minimum_across_tasks=(
                "number_models_generated",
                "min",
            ),
            maximum_across_tasks=(
                "number_models_generated",
                "max",
            ),
            number_of_tasks=(
                "task",
                "nunique",
            ),
            epsilon=(
                "epsilon",
                "first",
            ),
            lambda_value=(
                "lambda",
                "first",
            ),
            omega=(
                "omega",
                "first",
            ),
            sigma=(
                "sigma",
                "first",
            ),
        )
        .rename(
            columns={
                "lambda_value": "lambda",
            }
        )
    )

    subject_summary[
        "mean_number_models_generated"
    ] = subject_summary[
        "mean_number_models_generated"
    ].round(4)

    subject_summary[
        "std_across_tasks"
    ] = subject_summary[
        "std_across_tasks"
    ].round(4)

    subject_summary.to_csv(
        SUBJECT_SUMMARY_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80)

    print(
        "Fichier détaillé :",
        DETAILED_OUTPUT_FILE,
    )

    print(
        "Moyenne par participant :",
        SUBJECT_SUMMARY_FILE,
    )

    print(
        "Cache post-hoc :",
        POSTHOC_CACHE_FILE,
    )

    print(
        "\nNombre de lignes détaillées :",
        len(detailed_dataframe),
    )

    expected_number_of_rows = (
        len(subject_logs)
        * len(unique_tasks)
    )

    print(
        "Nombre de lignes attendu :",
        expected_number_of_rows,
    )

    print("\nPremières lignes détaillées :")

    print(
        detailed_dataframe
        .head(12)
        .to_string(index=False)
    )

    print("\nPremières moyennes par participant :")

    print(
        subject_summary
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
