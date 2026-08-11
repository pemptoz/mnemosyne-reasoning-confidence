"""
count_mental_model_E2.py

Analyse post-hoc du nombre de modèles générés par mReasoner pour les
phases intuitive et réfléchie du dataset E2.

Le script :

1. vérifie que dataset_ccobra_E2_int.csv et
   dataset_ccobra_E2_ref.csv contiennent les mêmes essais ;
2. utilise dataset_ccobra_E2_ref.csv comme source commune des données ;
3. charge les paramètres ajustés sur les réponses intuitives depuis
   log_full_E2_int.json ;
4. charge les paramètres ajustés sur les réponses réfléchies depuis
   log_full_E2_ref.json ;
5. interroge mReasoner avec les deux jeux de paramètres ;
6. compte les occurrences de #<Q-MODEL dans les traces Lisp ;
7. produit deux estimations distinctes :
       number_models_generated_int
       number_models_generated_ref
8. calcule :
       models_change = réfléchi - intuitif
9. produit :
       mental_models_E2.csv
       mental_models_subject_summary_E2.csv
       mental_models_task_summary_E2.csv

Aucune classification Système 1 / Système 2 n'est effectuée.

Le cache post-hoc dépend uniquement :
    - des prémisses ;
    - des paramètres mReasoner ;
    - du nombre de simulations.

Il peut donc être partagé entre les ajustements intuitif et réfléchi.
"""

import ast
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path



# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "pymreasoner_2",
)

if MODEL_DIR not in sys.path:
    sys.path.insert(
        0,
        MODEL_DIR,
    )

import mreasoner


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mreasoner


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

DATASET_FILE_INT = str(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_ccobra_E2_int.csv"
)

DATASET_FILE_REF = str(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_ccobra_E2_ref.csv"
)

DATASET_FILE = DATASET_FILE_REF


# ----------------------------------------------------------------------
# Journaux de paramètres
# ----------------------------------------------------------------------

PARAMETER_LOG_FILE_INT = str(
    PROJECT_ROOT
    / "results"
    / "logs"
    / "log_full_E2_int.json"
)

PARAMETER_LOG_FILE_REF = str(
    PROJECT_ROOT
    / "results"
    / "logs"
    / "log_full_E2_ref.json"
)


# ----------------------------------------------------------------------
# Chemins mReasoner
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Fichiers produits
# ----------------------------------------------------------------------

DETAILED_OUTPUT_FILE = str(
    PROJECT_ROOT
    / "results"
    / "tables"
    / "mental_models"
    / "mental_models_E2.csv"
)

SUBJECT_SUMMARY_FILE = str(
    PROJECT_ROOT
    / "results"
    / "tables"
    / "mental_models"
    / "mental_models_subject_summary_E2.csv"
)

TASK_SUMMARY_FILE = str(
    PROJECT_ROOT
    / "results"
    / "tables"
    / "mental_models"
    / "mental_models_task_summary_E2.csv"
)

POSTHOC_CACHE_FILE = str(
    PROJECT_ROOT
    / "data"
    / "cache"
    / "mental_models_posthoc_cache_E2.json"
)



# ----------------------------------------------------------------------
# Paramètres de simulation
# ----------------------------------------------------------------------

# Nombre de simulations stochastiques par combinaison :
#
#     tâche × paramètres
#
# Utiliser 3 pour tester.
# Utiliser 10 ou 20 pour une estimation plus stable.
N_SAMPLES = 3

# "selected" :
#     utilise les paramètres sélectionnés directement dans le journal.
#
# "all_best" :
#     utilise toutes les configurations de best_params et agrège leurs
#     simulations.
PARAMETER_SELECTION = "selected"

# Sauvegarde périodique du cache.
SAVE_CACHE_EVERY = 20

# Affichage de quelques traces Lisp pour contrôler le comptage.
DEBUG_TRACES = True
MAX_DEBUG_TRACES = 3


# ======================================================================
# OUTILS JSON
# ======================================================================

def atomic_json_dump(data, output_path):
    """
    Sauvegarde un objet JSON de manière atomique.
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
        prefix=".mental-models-",
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
            os.remove(
                temporary_path
            )


# ======================================================================
# NORMALISATION GÉNÉRALE
# ======================================================================

def normalize_subject_id(value):
    """
    Normalise l'identifiant d'un participant.

    Exemples :
        63873     -> "63873"
        63873.0   -> "63873"
        "63873"   -> "63873"
    """
    if pd.isna(value):
        return None

    normalized = str(value).strip()

    if not normalized:
        return None

    try:
        numeric = float(normalized)

        if numeric.is_integer():
            return str(
                int(numeric)
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return normalized


def parse_possible_collection(value):
    """
    Convertit si possible une chaîne contenant une collection Python
    ou JSON.

    Exemples :
        "[['Yes']]"  -> [["Yes"]]
        '["A", "B"]' -> ["A", "B"]
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped:
        return value

    if not stripped.startswith(
        ("[", "(", "{")
    ):
        return value

    try:
        return json.loads(
            stripped
        )

    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(
            stripped
        )

    except (
        SyntaxError,
        ValueError,
    ):
        return value


def unwrap_singletons(value):
    """
    Retire récursivement les collections contenant un seul élément.

    Exemples :
        [["Yes"]] -> "Yes"
        ["No"]    -> "No"
    """
    value = parse_possible_collection(
        value
    )

    while isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        if np.size(value) != 1:
            break

        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]

        else:
            value = value[0]

    return value


def normalize_yes_no(value):
    """
    Convertit une réponse vers Yes ou No.
    """
    if pd.isna(value):
        return np.nan

    value = unwrap_singletons(
        value
    )

    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"

    normalized = str(value).strip().lower()

    yes_values = {
        "yes",
        "y",
        "oui",
        "true",
        "1",
        "1.0",
        "valid",
        "follows",
    }

    no_values = {
        "no",
        "n",
        "non",
        "false",
        "0",
        "0.0",
        "invalid",
        "does not follow",
        "doesn't follow",
        "nvc",
    }

    if normalized in yes_values:
        return "Yes"

    if normalized in no_values:
        return "No"

    return np.nan


def normalize_binary(value):
    """
    Convertit une valeur binaire vers 0 ou 1.
    """
    if pd.isna(value):
        return np.nan

    value = unwrap_singletons(
        value
    )

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        numeric_value = float(value)

        if numeric_value == 1:
            return 1

        if numeric_value == 0:
            return 0

    normalized = str(value).strip().lower()

    true_values = {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "correct",
        "valid",
        "believable",
        "conflict",
    }

    false_values = {
        "0",
        "0.0",
        "false",
        "no",
        "n",
        "incorrect",
        "invalid",
        "unbelievable",
        "no conflict",
        "no-conflict",
        "non-conflict",
    }

    if normalized in true_values:
        return 1

    if normalized in false_values:
        return 0

    return np.nan


# ======================================================================
# NORMALISATION DES TÂCHES
# ======================================================================

TASK_MAPPING = {
    "MP": (
        "All B are C",
        "All A are B",
    ),
    "MT": (
        "All B are C",
        "No A are C",
    ),
    "AC": (
        "All B are C",
        "All A are C",
    ),
    "DA": (
        "All B are C",
        "No A are B",
    ),
}


def normalize_task_type(value):
    """
    Normalise un type de tâche vers MP, MT, AC ou DA.
    """
    if pd.isna(value):
        return None

    normalized = str(value).strip().upper()

    if normalized in TASK_MAPPING:
        return normalized

    return None


def normalize_premises(value):
    """
    Extrait exactement deux prémisses depuis la colonne task.

    Gère notamment :
        "All B are C/All A are B"
        ["All B are C", "All A are B"]
        [["All B are C"], ["All A are B"]]
    """
    value = parse_possible_collection(
        value
    )

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        premises = []

        for premise in value:
            premise = unwrap_singletons(
                premise
            )

            if isinstance(premise, (list, tuple)):
                premise = " ".join(
                    str(part)
                    for part in premise
                )

            premise = str(premise).strip()

            if premise:
                premises.append(
                    premise
                )

    else:
        task_string = str(value).strip()

        if "/" in task_string:
            premises = [
                part.strip()
                for part in task_string.split("/")
                if part.strip()
            ]

        elif ";" in task_string:
            premises = [
                part.strip()
                for part in task_string.split(";")
                if part.strip()
            ]

        else:
            premises = []

    if len(premises) != 2:
        raise ValueError(
            "Impossible d'extraire exactement deux prémisses depuis "
            f"{value!r}. Résultat obtenu : {premises!r}"
        )

    return tuple(
        premises
    )


def infer_task_type(premises):
    """
    Déduit MP, MT, AC ou DA depuis les deux prémisses.
    """
    normalized_premises = tuple(
        str(premise).strip()
        for premise in premises
    )

    for task_type, expected_premises in TASK_MAPPING.items():
        if normalized_premises == expected_premises:
            return task_type

    return None


# ======================================================================
# PARAMÈTRES MREASONER
# ======================================================================

PARAMETER_NAMES = (
    "epsilon",
    "lambda",
    "omega",
    "sigma",
)


def normalize_params(params):
    """
    Extrait et convertit les quatre paramètres de mReasoner.
    """
    missing_parameters = [
        name
        for name in PARAMETER_NAMES
        if name not in params
    ]

    if missing_parameters:
        raise ValueError(
            "Paramètres mReasoner manquants : "
            f"{missing_parameters}. Entrée : {params!r}"
        )

    return {
        name: float(params[name])
        for name in PARAMETER_NAMES
    }


def params_to_tuple(params):
    """
    Produit une clé numérique stable pour les paramètres.
    """
    return tuple(
        round(
            float(params[name]),
            12,
        )
        for name in PARAMETER_NAMES
    )


def load_subject_logs(parameter_log_file):
    """
    Charge les paramètres individuels depuis un journal mReasoner.
    """
    if not os.path.isfile(
        parameter_log_file
    ):
        raise FileNotFoundError(
            "Fichier de paramètres introuvable : "
            f"{parameter_log_file}"
        )

    with open(
        parameter_log_file,
        "r",
        encoding="utf-8",
    ) as input_file:
        full_log = json.load(
            input_file
        )

    if not isinstance(full_log, dict):
        raise ValueError(
            "Le journal doit contenir un objet JSON : "
            f"{parameter_log_file}"
        )

    if "mReasoner" not in full_log:
        raise KeyError(
            "La clé 'mReasoner' est absente de "
            f"{parameter_log_file}."
        )

    subject_logs = full_log[
        "mReasoner"
    ]

    if not isinstance(subject_logs, dict):
        raise ValueError(
            "La section 'mReasoner' doit être un dictionnaire "
            f"dans {parameter_log_file}."
        )

    normalized_logs = {}

    for subject_id, subject_log in subject_logs.items():
        normalized_subject_id = normalize_subject_id(
            subject_id
        )

        if normalized_subject_id is None:
            continue

        normalized_logs[
            normalized_subject_id
        ] = subject_log

    return normalized_logs


def extract_parameter_sets(subject_log):
    """
    Retourne la ou les configurations de paramètres à utiliser.
    """
    selected_params = normalize_params(
        subject_log
    )

    if PARAMETER_SELECTION == "selected":
        return [selected_params]

    if PARAMETER_SELECTION != "all_best":
        raise ValueError(
            "PARAMETER_SELECTION doit valoir "
            "'selected' ou 'all_best'."
        )

    raw_best_params = subject_log.get(
        "best_params",
        [],
    )

    if not raw_best_params:
        return [selected_params]

    parameter_sets = []
    seen = set()

    for raw_params in raw_best_params:
        params = normalize_params(
            raw_params
        )

        key = params_to_tuple(
            params
        )

        if key in seen:
            continue

        seen.add(key)

        parameter_sets.append(
            params
        )

    if not parameter_sets:
        return [selected_params]

    return parameter_sets


# ======================================================================
# VÉRIFICATION DES DEUX DATASETS
# ======================================================================

def verify_phase_datasets():
    """
    Vérifie que les datasets intuitif et réfléchi contiennent exactement
    les mêmes essais dans le même ordre.

    Vérifie également :
        dataset intuitif : response == response_int
        dataset réfléchi : response == response_ref
    """
    if not os.path.isfile(
        DATASET_FILE_INT
    ):
        raise FileNotFoundError(
            "Dataset intuitif introuvable : "
            f"{DATASET_FILE_INT}"
        )

    if not os.path.isfile(
        DATASET_FILE_REF
    ):
        raise FileNotFoundError(
            "Dataset réfléchi introuvable : "
            f"{DATASET_FILE_REF}"
        )

    dataframe_int = pd.read_csv(
        DATASET_FILE_INT
    )

    dataframe_ref = pd.read_csv(
        DATASET_FILE_REF
    )

    required_columns = {
        "id",
        "sequence",
        "task",
        "response",
        "response_int",
        "response_ref",
    }

    missing_int = (
        required_columns
        - set(dataframe_int.columns)
    )

    missing_ref = (
        required_columns
        - set(dataframe_ref.columns)
    )

    if missing_int:
        raise KeyError(
            "Colonnes absentes du dataset intuitif : "
            f"{sorted(missing_int)}"
        )

    if missing_ref:
        raise KeyError(
            "Colonnes absentes du dataset réfléchi : "
            f"{sorted(missing_ref)}"
        )

    if len(dataframe_int) != len(dataframe_ref):
        raise ValueError(
            "Les datasets intuitif et réfléchi ne contiennent pas "
            "le même nombre d'essais : "
            f"{len(dataframe_int)} contre {len(dataframe_ref)}."
        )

    keys_int = pd.DataFrame({
        "id": dataframe_int["id"].apply(
            normalize_subject_id
        ),
        "sequence": pd.to_numeric(
            dataframe_int["sequence"],
            errors="coerce",
        ),
        "task": dataframe_int["task"].astype(str),
    })

    keys_ref = pd.DataFrame({
        "id": dataframe_ref["id"].apply(
            normalize_subject_id
        ),
        "sequence": pd.to_numeric(
            dataframe_ref["sequence"],
            errors="coerce",
        ),
        "task": dataframe_ref["task"].astype(str),
    })

    if not keys_int.equals(
        keys_ref
    ):
        raise ValueError(
            "Les datasets intuitif et réfléchi ne contiennent pas "
            "exactement les mêmes essais dans le même ordre."
        )

    response_int_target = (
        dataframe_int["response"]
        .apply(normalize_yes_no)
    )

    response_int_source = (
        dataframe_int["response_int"]
        .apply(normalize_yes_no)
    )

    response_ref_target = (
        dataframe_ref["response"]
        .apply(normalize_yes_no)
    )

    response_ref_source = (
        dataframe_ref["response_ref"]
        .apply(normalize_yes_no)
    )

    intuitive_matches = (
        response_int_target
        == response_int_source
    )

    reflective_matches = (
        response_ref_target
        == response_ref_source
    )

    if not intuitive_matches.all():
        mismatch_count = int(
            (~intuitive_matches).sum()
        )

        raise ValueError(
            "Dans le dataset intuitif, response ne correspond pas "
            f"à response_int pour {mismatch_count} essai(s)."
        )

    if not reflective_matches.all():
        mismatch_count = int(
            (~reflective_matches).sum()
        )

        raise ValueError(
            "Dans le dataset réfléchi, response ne correspond pas "
            f"à response_ref pour {mismatch_count} essai(s)."
        )

    changed_responses = int(
        (
            response_int_target
            != response_ref_target
        ).sum()
    )

    print(
        "Datasets intuitif et réfléchi vérifiés :",
        len(dataframe_int),
        "essais identiques.",
    )

    print(
        "Essais avec changement de réponse :",
        changed_responses,
    )


# ======================================================================
# CHARGEMENT DU DATASET COMMUN
# ======================================================================

def load_trials():
    """
    Charge dataset_ccobra_E2_ref.csv comme source commune des essais.

    Les paramètres intuitifs et réfléchis proviennent des deux journaux
    séparés, et non de la colonne response de ce fichier.
    """
    if not os.path.isfile(
        DATASET_FILE
    ):
        raise FileNotFoundError(
            "Dataset E2 introuvable : "
            f"{DATASET_FILE}"
        )

    dataframe = pd.read_csv(
        DATASET_FILE
    )

    required_columns = {
        "id",
        "sequence",
        "task",
        "response_int",
        "response_ref",
        "correct_int",
        "correct_ref",
        "for_int",
        "for_ref",
        "rt_int",
        "rt_ref",
        "validity",
        "believability",
        "conflict",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Colonnes absentes du dataset E2 : "
            f"{sorted(missing_columns)}.\n"
            f"Colonnes disponibles : {list(dataframe.columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["subject_id"] = (
        dataframe["id"]
        .apply(normalize_subject_id)
    )

    dataframe["sequence"] = pd.to_numeric(
        dataframe["sequence"],
        errors="coerce",
    )

    dataframe["premises"] = (
        dataframe["task"]
        .apply(normalize_premises)
    )

    if "task_type" in dataframe.columns:
        dataframe["task_type"] = (
            dataframe["task_type"]
            .apply(normalize_task_type)
        )

    else:
        dataframe["task_type"] = (
            dataframe["premises"]
            .apply(infer_task_type)
        )

    dataframe["response_int"] = (
        dataframe["response_int"]
        .apply(normalize_yes_no)
    )

    dataframe["response_ref"] = (
        dataframe["response_ref"]
        .apply(normalize_yes_no)
    )

    for column in [
        "correct_int",
        "correct_ref",
        "validity",
        "believability",
        "conflict",
    ]:
        dataframe[column] = (
            dataframe[column]
            .apply(normalize_binary)
        )

    for column in [
        "for_int",
        "for_ref",
        "rt_int",
        "rt_ref",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_after_normalization = [
        "subject_id",
        "sequence",
        "premises",
        "task_type",
        "response_int",
        "response_ref",
        "correct_int",
        "correct_ref",
        "validity",
        "believability",
        "conflict",
    ]

    before_drop = len(
        dataframe
    )

    dataframe = (
        dataframe
        .dropna(
            subset=required_after_normalization
        )
        .copy()
    )

    removed_rows = (
        before_drop - len(dataframe)
    )

    if removed_rows:
        print(
            "Essais supprimés car incomplets :",
            removed_rows,
        )

    dataframe["sequence"] = (
        dataframe["sequence"]
        .astype(int)
    )

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "subject_id",
                "sequence",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "Essais chargés :",
        len(dataframe),
    )

    print(
        "Participants chargés :",
        dataframe["subject_id"].nunique(),
    )

    print(
        "Précision intuitive :",
        round(
            dataframe["correct_int"].mean()
            * 100,
            4,
        ),
        "%",
    )

    print(
        "Précision réfléchie :",
        round(
            dataframe["correct_ref"].mean()
            * 100,
            4,
        ),
        "%",
    )

    return dataframe


# ======================================================================
# TRACE ET COMPTAGE DES MODÈLES
# ======================================================================

def parse_models_count(trace_string):
    """
    Compte le nombre d'occurrences de #<Q-MODEL dans une trace Lisp.
    """
    if trace_string is None:
        return 0

    normalized_trace = str(
        trace_string
    )

    if not normalized_trace.strip():
        return 0

    return normalized_trace.upper().count(
        "#<Q-MODEL"
    )


# ======================================================================
# CACHE POST-HOC
# ======================================================================

def load_posthoc_cache():
    """
    Charge le cache des simulations post-hoc.
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
            "Cache post-hoc chargé :",
            len(cache),
            "entrée(s).",
        )

        return cache

    except (
        OSError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            "WARNING: cache post-hoc illisible :",
            repr(error),
        )

        return {}


def make_cache_key(
    premises,
    params,
):
    """
    Construit une clé stable pour une combinaison :

        prémisses + paramètres + nombre de simulations.
    """
    epsilon, lambda_, omega, sigma = (
        params_to_tuple(params)
    )

    payload = {
        "cache_version": 3,
        "count_rule": "count_Q_MODEL_only",
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
# SIMULATION D'UNE TÂCHE
# ======================================================================

def run_trace_query(
    mr,
    premises,
    params,
):
    """
    Lance query_trace() et compte le nombre de modèles générés.
    """
    trace_string = mr.query_trace(
        list(premises),
        param_dict=params,
    )

    model_count = parse_models_count(
        trace_string
    )

    return {
        "trace": trace_string,
        "model_count": model_count,
    }


def compute_task_statistics(
    mr,
    premises,
    params,
    posthoc_cache,
    debug_state,
):
    """
    Calcule le nombre de modèles générés pour une combinaison de
    prémisses et de paramètres.
    """
    cache_key = make_cache_key(
        premises=premises,
        params=params,
    )

    if cache_key in posthoc_cache:
        cached = posthoc_cache[
            cache_key
        ]

        sample_counts = cached.get(
            "sample_counts",
            [],
        )

        if sample_counts:
            return {
                "sample_counts": [
                    int(value)
                    for value in sample_counts
                ],
                "from_cache": True,
            }

    sample_counts = []

    for sample_index in range(
        N_SAMPLES
    ):
        query_result = run_trace_query(
            mr=mr,
            premises=premises,
            params=params,
        )

        model_count = query_result[
            "model_count"
        ]

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
                "Simulation :",
                sample_index + 1,
                "/",
                N_SAMPLES,
            )

            print(
                "Nombre de modèles générés :",
                model_count,
            )

            print("Trace brute :")
            print(
                query_result["trace"]
            )

            print("=" * 80)

            debug_state["printed"] += 1

    posthoc_cache[
        cache_key
    ] = {
        "premises": list(premises),
        "parameters": {
            name: float(params[name])
            for name in PARAMETER_NAMES
        },
        "n_samples": int(N_SAMPLES),
        "sample_counts": [
            int(value)
            for value in sample_counts
        ],
    }

    return {
        "sample_counts": sample_counts,
        "from_cache": False,
    }


# ======================================================================
# AGRÉGATION DES SIMULATIONS
# ======================================================================

def summarize_simulations(
    simulation_results,
):
    """
    Agrège les nombres de modèles obtenus avec une ou plusieurs
    configurations de paramètres.
    """
    all_counts = []

    for result in simulation_results:
        all_counts.extend(
            result["sample_counts"]
        )

    if not all_counts:
        raise RuntimeError(
            "Aucune simulation n'a été produite."
        )

    count_array = np.asarray(
        all_counts,
        dtype=np.float64,
    )

    return {
        "number_models_generated":
            float(np.mean(count_array)),

        "std_models_generated":
            float(np.std(count_array, ddof=0)),

        "minimum_models_generated":
            int(np.min(count_array)),

        "maximum_models_generated":
            int(np.max(count_array)),

        "total_simulation_count":
            int(len(count_array)),
    }


def simulate_parameter_sets(
    mr,
    premises,
    parameter_sets,
    posthoc_cache,
    debug_state,
):
    """
    Interroge mReasoner avec toutes les configurations de paramètres
    fournies, puis agrège le nombre de modèles générés.

    Retourne :
        summary :
            statistiques agrégées ;

        new_cache_entries :
            nombre de nouvelles entrées ajoutées au cache.
    """
    simulation_results = []
    new_cache_entries = 0

    for params in parameter_sets:
        cache_key = make_cache_key(
            premises=premises,
            params=params,
        )

        was_cached = (
            cache_key in posthoc_cache
        )

        simulation_result = compute_task_statistics(
            mr=mr,
            premises=premises,
            params=params,
            posthoc_cache=posthoc_cache,
            debug_state=debug_state,
        )

        simulation_results.append(
            simulation_result
        )

        if not was_cached:
            new_cache_entries += 1

    summary = summarize_simulations(
        simulation_results=
            simulation_results
    )

    return summary, new_cache_entries


# ======================================================================
# RÉSUMÉS
# ======================================================================

def build_subject_summary(detailed_dataframe):
    """
    Construit le résumé par participant.
    """
    subject_summary = (
        detailed_dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            mean_number_models_generated_int=(
                "number_models_generated_int",
                "mean",
            ),

            median_number_models_generated_int=(
                "number_models_generated_int",
                "median",
            ),

            std_number_models_generated_int=(
                "number_models_generated_int",
                "std",
            ),

            minimum_number_models_generated_int=(
                "minimum_models_generated_int",
                "min",
            ),

            maximum_number_models_generated_int=(
                "maximum_models_generated_int",
                "max",
            ),

            mean_number_models_generated_ref=(
                "number_models_generated_ref",
                "mean",
            ),

            median_number_models_generated_ref=(
                "number_models_generated_ref",
                "median",
            ),

            std_number_models_generated_ref=(
                "number_models_generated_ref",
                "std",
            ),

            minimum_number_models_generated_ref=(
                "minimum_models_generated_ref",
                "min",
            ),

            maximum_number_models_generated_ref=(
                "maximum_models_generated_ref",
                "max",
            ),

            mean_models_change=(
                "models_change",
                "mean",
            ),

            intuitive_accuracy=(
                "correct_int",
                "mean",
            ),

            reflective_accuracy=(
                "correct_ref",
                "mean",
            ),

            mean_for_int=(
                "for_int",
                "mean",
            ),

            mean_for_ref=(
                "for_ref",
                "mean",
            ),

            mean_rt_int=(
                "rt_int",
                "mean",
            ),

            mean_rt_ref=(
                "rt_ref",
                "mean",
            ),

            response_change_rate=(
                "response_changed",
                "mean",
            ),

            mean_accuracy_gain=(
                "accuracy_gain",
                "mean",
            ),

            mean_for_change=(
                "for_change",
                "mean",
            ),

            mean_rt_change=(
                "rt_change",
                "mean",
            ),

            number_of_trials=(
                "sequence",
                "count",
            ),

            number_of_task_types=(
                "task_type",
                "nunique",
            ),

            epsilon_int=(
                "epsilon_int",
                "first",
            ),

            lambda_int=(
                "lambda_int",
                "first",
            ),

            omega_int=(
                "omega_int",
                "first",
            ),

            sigma_int=(
                "sigma_int",
                "first",
            ),

            epsilon_ref=(
                "epsilon_ref",
                "first",
            ),

            lambda_ref=(
                "lambda_ref",
                "first",
            ),

            omega_ref=(
                "omega_ref",
                "first",
            ),

            sigma_ref=(
                "sigma_ref",
                "first",
            ),
        )
    )

    for percentage_column in [
        "intuitive_accuracy",
        "reflective_accuracy",
        "response_change_rate",
        "mean_accuracy_gain",
    ]:
        subject_summary[
            percentage_column
        ] *= 100

    subject_summary = (
        subject_summary
        .sort_values(
            by="subject_id"
        )
        .reset_index(
            drop=True
        )
    )

    return subject_summary


def build_task_summary(detailed_dataframe):
    """
    Construit le résumé par type de tâche.
    """
    task_summary = (
        detailed_dataframe
        .groupby(
            "task_type",
            as_index=False,
        )
        .agg(
            mean_number_models_generated_int=(
                "number_models_generated_int",
                "mean",
            ),

            median_number_models_generated_int=(
                "number_models_generated_int",
                "median",
            ),

            std_number_models_generated_int=(
                "number_models_generated_int",
                "std",
            ),

            minimum_number_models_generated_int=(
                "minimum_models_generated_int",
                "min",
            ),

            maximum_number_models_generated_int=(
                "maximum_models_generated_int",
                "max",
            ),

            mean_number_models_generated_ref=(
                "number_models_generated_ref",
                "mean",
            ),

            median_number_models_generated_ref=(
                "number_models_generated_ref",
                "median",
            ),

            std_number_models_generated_ref=(
                "number_models_generated_ref",
                "std",
            ),

            minimum_number_models_generated_ref=(
                "minimum_models_generated_ref",
                "min",
            ),

            maximum_number_models_generated_ref=(
                "maximum_models_generated_ref",
                "max",
            ),

            mean_models_change=(
                "models_change",
                "mean",
            ),

            intuitive_accuracy=(
                "correct_int",
                "mean",
            ),

            reflective_accuracy=(
                "correct_ref",
                "mean",
            ),

            mean_for_int=(
                "for_int",
                "mean",
            ),

            mean_for_ref=(
                "for_ref",
                "mean",
            ),

            mean_rt_int=(
                "rt_int",
                "mean",
            ),

            mean_rt_ref=(
                "rt_ref",
                "mean",
            ),

            response_change_rate=(
                "response_changed",
                "mean",
            ),

            mean_accuracy_gain=(
                "accuracy_gain",
                "mean",
            ),

            number_of_trials=(
                "sequence",
                "count",
            ),

            number_of_subjects=(
                "subject_id",
                "nunique",
            ),
        )
    )

    for percentage_column in [
        "intuitive_accuracy",
        "reflective_accuracy",
        "response_change_rate",
        "mean_accuracy_gain",
    ]:
        task_summary[
            percentage_column
        ] *= 100

    task_summary = (
        task_summary
        .sort_values(
            by="task_type"
        )
        .reset_index(
            drop=True
        )
    )

    return task_summary


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print(
        "ANALYSE POST-HOC MREASONER — "
        "AJUSTEMENTS INTUITIF ET RÉFLÉCHI"
    )
    print("=" * 80)

    print(
        "Dataset intuitif :",
        DATASET_FILE_INT,
    )

    print(
        "Dataset réfléchi :",
        DATASET_FILE_REF,
    )

    print(
        "Journal des paramètres intuitifs :",
        PARAMETER_LOG_FILE_INT,
    )

    print(
        "Journal des paramètres réfléchis :",
        PARAMETER_LOG_FILE_REF,
    )

    print(
        "Nombre de simulations par configuration :",
        N_SAMPLES,
    )

    print(
        "Sélection des paramètres :",
        PARAMETER_SELECTION,
    )

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


    # Vérification des deux fichiers expérimentaux.
    verify_phase_datasets()

    # Chargement des deux journaux.
    subject_logs_int = load_subject_logs(
        PARAMETER_LOG_FILE_INT
    )

    subject_logs_ref = load_subject_logs(
        PARAMETER_LOG_FILE_REF
    )

    trials = load_trials()
    posthoc_cache = load_posthoc_cache()

    # ------------------------------------------------------------------
    # Participants communs
    # ------------------------------------------------------------------

    subject_ids_in_dataset = set(
        trials["subject_id"]
    )

    subject_ids_in_log_int = set(
        subject_logs_int
    )

    subject_ids_in_log_ref = set(
        subject_logs_ref
    )

    common_subject_ids = (
        subject_ids_in_dataset
        & subject_ids_in_log_int
        & subject_ids_in_log_ref
    )

    missing_in_log_int = sorted(
        subject_ids_in_dataset
        - subject_ids_in_log_int
    )

    missing_in_log_ref = sorted(
        subject_ids_in_dataset
        - subject_ids_in_log_ref
    )

    missing_in_dataset_int = sorted(
        subject_ids_in_log_int
        - subject_ids_in_dataset
    )

    missing_in_dataset_ref = sorted(
        subject_ids_in_log_ref
        - subject_ids_in_dataset
    )

    print(
        "\nParticipants dans le dataset :",
        len(subject_ids_in_dataset),
    )

    print(
        "Participants dans le journal intuitif :",
        len(subject_ids_in_log_int),
    )

    print(
        "Participants dans le journal réfléchi :",
        len(subject_ids_in_log_ref),
    )

    print(
        "Participants communs aux trois sources :",
        len(common_subject_ids),
    )

    if missing_in_log_int:
        print(
            "WARNING: participants sans paramètres intuitifs :",
            missing_in_log_int[:10],
        )

    if missing_in_log_ref:
        print(
            "WARNING: participants sans paramètres réfléchis :",
            missing_in_log_ref[:10],
        )

    if missing_in_dataset_int:
        print(
            "WARNING: participants du journal intuitif "
            "absents du dataset :",
            missing_in_dataset_int[:10],
        )

    if missing_in_dataset_ref:
        print(
            "WARNING: participants du journal réfléchi "
            "absents du dataset :",
            missing_in_dataset_ref[:10],
        )

    if not common_subject_ids:
        raise RuntimeError(
            "Aucun participant commun entre le dataset et les deux "
            "journaux de paramètres."
        )

    trials = trials.loc[
        trials["subject_id"].isin(
            common_subject_ids
        )
    ].copy()

    print(
        "Nombre d'essais à analyser :",
        len(trials),
    )

    # ------------------------------------------------------------------
    # Initialisation de mReasoner
    # ------------------------------------------------------------------

    print(
        "\nDémarrage de mReasoner..."
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
    last_cache_save_count = 0

    try:
        grouped_trials = trials.groupby(
            "subject_id",
            sort=True,
        )

        participant_iterator = tqdm(
            grouped_trials,
            total=trials["subject_id"].nunique(),
            desc="Participants",
        )

        for subject_id, participant_trials in participant_iterator:
            # ----------------------------------------------------------
            # Paramètres intuitifs
            # ----------------------------------------------------------

            subject_log_int = subject_logs_int[
                subject_id
            ]

            selected_params_int = normalize_params(
                subject_log_int
            )

            parameter_sets_int = extract_parameter_sets(
                subject_log_int
            )

            # ----------------------------------------------------------
            # Paramètres réfléchis
            # ----------------------------------------------------------

            subject_log_ref = subject_logs_ref[
                subject_id
            ]

            selected_params_ref = normalize_params(
                subject_log_ref
            )

            parameter_sets_ref = extract_parameter_sets(
                subject_log_ref
            )

            # ----------------------------------------------------------
            # Essais du participant
            # ----------------------------------------------------------

            for _, trial in participant_trials.iterrows():
                premises = trial[
                    "premises"
                ]

                # Paramètres ajustés sur les réponses intuitives.
                summary_int, new_entries_int = (
                    simulate_parameter_sets(
                        mr=mr,
                        premises=premises,
                        parameter_sets=parameter_sets_int,
                        posthoc_cache=posthoc_cache,
                        debug_state=debug_state,
                    )
                )

                new_cache_entries += (
                    new_entries_int
                )

                # Paramètres ajustés sur les réponses réfléchies.
                summary_ref, new_entries_ref = (
                    simulate_parameter_sets(
                        mr=mr,
                        premises=premises,
                        parameter_sets=parameter_sets_ref,
                        posthoc_cache=posthoc_cache,
                        debug_state=debug_state,
                    )
                )

                new_cache_entries += (
                    new_entries_ref
                )

                # Sauvegarde périodique du cache.
                if (
                    new_cache_entries
                    - last_cache_save_count
                    >= SAVE_CACHE_EVERY
                ):
                    atomic_json_dump(
                        posthoc_cache,
                        POSTHOC_CACHE_FILE,
                    )

                    last_cache_save_count = (
                        new_cache_entries
                    )

                # ------------------------------------------------------
                # Mesures expérimentales
                # ------------------------------------------------------

                response_changed = int(
                    trial["response_int"]
                    != trial["response_ref"]
                )

                accuracy_gain = int(
                    trial["correct_ref"]
                    - trial["correct_int"]
                )

                for_change = (
                    trial["for_ref"]
                    - trial["for_int"]
                    if (
                        pd.notna(trial["for_ref"])
                        and pd.notna(trial["for_int"])
                    )
                    else np.nan
                )

                rt_change = (
                    trial["rt_ref"]
                    - trial["rt_int"]
                    if (
                        pd.notna(trial["rt_ref"])
                        and pd.notna(trial["rt_int"])
                    )
                    else np.nan
                )

                # ------------------------------------------------------
                # Ligne détaillée
                # ------------------------------------------------------

                detailed_results.append({
                    "subject_id":
                        subject_id,

                    "sequence":
                        int(trial["sequence"]),

                    "task_type":
                        trial["task_type"],

                    "premise_1":
                        premises[0],

                    "premise_2":
                        premises[1],

                    "validity":
                        int(trial["validity"]),

                    "believability":
                        int(trial["believability"]),

                    "conflict":
                        int(trial["conflict"]),

                    "response_int":
                        trial["response_int"],

                    "response_ref":
                        trial["response_ref"],

                    "correct_int":
                        int(trial["correct_int"]),

                    "correct_ref":
                        int(trial["correct_ref"]),

                    "for_int":
                        trial["for_int"],

                    "for_ref":
                        trial["for_ref"],

                    "rt_int":
                        trial["rt_int"],

                    "rt_ref":
                        trial["rt_ref"],

                    "response_changed":
                        response_changed,

                    "accuracy_gain":
                        accuracy_gain,

                    "for_change":
                        for_change,

                    "rt_change":
                        rt_change,

                    # Paramètres intuitifs.
                    "epsilon_int":
                        selected_params_int["epsilon"],

                    "lambda_int":
                        selected_params_int["lambda"],

                    "omega_int":
                        selected_params_int["omega"],

                    "sigma_int":
                        selected_params_int["sigma"],

                    "n_parameter_sets_used_int":
                        len(parameter_sets_int),

                    # Paramètres réfléchis.
                    "epsilon_ref":
                        selected_params_ref["epsilon"],

                    "lambda_ref":
                        selected_params_ref["lambda"],

                    "omega_ref":
                        selected_params_ref["omega"],

                    "sigma_ref":
                        selected_params_ref["sigma"],

                    "n_parameter_sets_used_ref":
                        len(parameter_sets_ref),

                    # Modèles avec les paramètres intuitifs.
                    "number_models_generated_int":
                        summary_int[
                            "number_models_generated"
                        ],

                    "std_models_generated_int":
                        summary_int[
                            "std_models_generated"
                        ],

                    "minimum_models_generated_int":
                        summary_int[
                            "minimum_models_generated"
                        ],

                    "maximum_models_generated_int":
                        summary_int[
                            "maximum_models_generated"
                        ],

                    "total_simulation_count_int":
                        summary_int[
                            "total_simulation_count"
                        ],

                    # Modèles avec les paramètres réfléchis.
                    "number_models_generated_ref":
                        summary_ref[
                            "number_models_generated"
                        ],

                    "std_models_generated_ref":
                        summary_ref[
                            "std_models_generated"
                        ],

                    "minimum_models_generated_ref":
                        summary_ref[
                            "minimum_models_generated"
                        ],

                    "maximum_models_generated_ref":
                        summary_ref[
                            "maximum_models_generated"
                        ],

                    "total_simulation_count_ref":
                        summary_ref[
                            "total_simulation_count"
                        ],

                    # Différence : réfléchi moins intuitif.
                    "models_change":
                        (
                            summary_ref[
                                "number_models_generated"
                            ]
                            - summary_int[
                                "number_models_generated"
                            ]
                        ),
                })

        # Sauvegarde finale du cache.
        atomic_json_dump(
            posthoc_cache,
            POSTHOC_CACHE_FILE,
        )

    finally:
        print(
            "\nFermeture de mReasoner..."
        )

        try:
            mr.terminate()

        except Exception as error:
            print(
                "WARNING: impossible de fermer mReasoner :",
                repr(error),
            )

    if not detailed_results:
        raise RuntimeError(
            "Aucun résultat détaillé n'a été produit."
        )

    # ------------------------------------------------------------------
    # Fichier détaillé
    # ------------------------------------------------------------------

    detailed_dataframe = pd.DataFrame(
        detailed_results
    )

    detailed_dataframe = (
        detailed_dataframe
        .sort_values(
            by=[
                "subject_id",
                "sequence",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    detailed_dataframe.to_csv(
        DETAILED_OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Résumé par participant
    # ------------------------------------------------------------------

    subject_summary = build_subject_summary(
        detailed_dataframe
    )

    subject_summary.to_csv(
        SUBJECT_SUMMARY_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Résumé par type de tâche
    # ------------------------------------------------------------------

    task_summary = build_task_summary(
        detailed_dataframe
    )

    task_summary.to_csv(
        TASK_SUMMARY_FILE,
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
        "Résumé par participant :",
        SUBJECT_SUMMARY_FILE,
    )

    print(
        "Résumé par tâche :",
        TASK_SUMMARY_FILE,
    )

    print(
        "Cache post-hoc :",
        POSTHOC_CACHE_FILE,
    )

    print(
        "\nNombre d'essais analysés :",
        len(detailed_dataframe),
    )

    print(
        "Nombre de participants :",
        detailed_dataframe[
            "subject_id"
        ].nunique(),
    )

    print(
        "\nPrécision intuitive globale :",
        detailed_dataframe[
            "correct_int"
        ].mean() * 100,
        "%",
    )

    print(
        "Précision réfléchie globale :",
        detailed_dataframe[
            "correct_ref"
        ].mean() * 100,
        "%",
    )

    print(
        "\nNombre moyen de modèles — ajustement intuitif :",
        detailed_dataframe[
            "number_models_generated_int"
        ].mean(),
    )

    print(
        "Nombre moyen de modèles — ajustement réfléchi :",
        detailed_dataframe[
            "number_models_generated_ref"
        ].mean(),
    )

    print(
        "Différence moyenne réfléchi − intuitif :",
        detailed_dataframe[
            "models_change"
        ].mean(),
    )

    print(
        "\nNombre médian de modèles — ajustement intuitif :",
        detailed_dataframe[
            "number_models_generated_int"
        ].median(),
    )

    print(
        "Nombre médian de modèles — ajustement réfléchi :",
        detailed_dataframe[
            "number_models_generated_ref"
        ].median(),
    )

    print(
        "\nMinimum observé — ajustement intuitif :",
        detailed_dataframe[
            "minimum_models_generated_int"
        ].min(),
    )

    print(
        "Maximum observé — ajustement intuitif :",
        detailed_dataframe[
            "maximum_models_generated_int"
        ].max(),
    )

    print(
        "Minimum observé — ajustement réfléchi :",
        detailed_dataframe[
            "minimum_models_generated_ref"
        ].min(),
    )

    print(
        "Maximum observé — ajustement réfléchi :",
        detailed_dataframe[
            "maximum_models_generated_ref"
        ].max(),
    )

    print(
        "\nNombre moyen de modèles par type de tâche :"
    )

    print(
        detailed_dataframe
        .groupby(
            "task_type"
        )[[
            "number_models_generated_int",
            "number_models_generated_ref",
            "models_change",
        ]]
        .mean()
        .to_string()
    )

    print(
        "\nRésumé par type de tâche :"
    )

    print(
        task_summary.to_string(
            index=False
        )
    )

    print(
        "\nAperçu du fichier détaillé :"
    )

    print(
        detailed_dataframe
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
