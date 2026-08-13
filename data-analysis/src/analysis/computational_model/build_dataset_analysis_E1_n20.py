"""
build_dataset_analysis_E1.py

Construit le dataset analytique minimal utilisé par les modèles E1.

Entrées
-------
dataset_ccobra_E1.csv
mental_models_count_E1_n20.csv

Sorties normales
----------------
dataset_analysis_E1_n20.csv
analysis_E1_outputs/data_audit_E1.txt

Sorties créées uniquement en cas d'erreur
-----------------------------------------
analysis_E1_outputs/model_merge_errors_E1.csv
analysis_E1_outputs/item_consistency_errors_E1.csv
analysis_E1_outputs/model_structure_errors_E1.csv

Structure des données
---------------------
dataset_ccobra_E1.csv :
    une ligne = participant × essai

mental_models_count_E1_n20.csv :
    une ligne = participant × type de tâche

La fusion est effectuée sur :
    subject_id + task_type
"""

import math
import os
import re
import sys

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "..",
    )
)

EXPERIMENT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "dataset_ccobra_E1.csv",
)

MODEL_COUNT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "tables",
    "computational_model",
    "mental_models_count_E1_n20.csv",
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "tables",
    "computational_model",
    "dataset_analysis_E1_n20.csv",
)

OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "results",
    "analysis",
    "computational_model",
    "analysis_E1_outputs",
)

AUDIT_REPORT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "data_audit_E1.txt",
)

MODEL_MERGE_ERRORS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "model_merge_errors_E1.csv",
)

ITEM_CONSISTENCY_ERRORS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "item_consistency_errors_E1.csv",
)

MODEL_STRUCTURE_ERRORS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "model_structure_errors_E1.csv",
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True,
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


ENTROPY_LOG_BASE = 2
MIN_RESPONSES_FOR_ENTROPY = 2
STRICT_VALIDATION = True


# ======================================================================
# TYPES DE TÂCHE
# ======================================================================

TASK_MAPPING = {
    (
        "all b are c",
        "all a are b",
    ): "MP",

    (
        "all b are c",
        "no a are c",
    ): "MT",

    (
        "all b are c",
        "all a are c",
    ): "AC",

    (
        "all b are c",
        "no a are b",
    ): "DA",
}

EXPECTED_TASK_TYPES = {
    "MP",
    "MT",
    "AC",
    "DA",
}


# ======================================================================
# JOURNAL D'AUDIT
# ======================================================================

AUDIT_LINES = []


def audit_print(*values):
    """Affiche un message et l'ajoute au rapport d'audit."""
    text = " ".join(
        str(value)
        for value in values
    )

    print(text)
    AUDIT_LINES.append(text)


def audit_section(title):
    """Ajoute un titre de section."""
    separator = "=" * 80

    audit_print("")
    audit_print(separator)
    audit_print(title)
    audit_print(separator)


def save_audit_report():
    """Enregistre le rapport d'audit."""
    with open(
        AUDIT_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "\n".join(AUDIT_LINES)
        )
        output_file.write("\n")

    print(
        "Rapport d'audit enregistré :",
        AUDIT_REPORT_FILE,
    )


# ======================================================================
# NORMALISATION
# ======================================================================

def normalize_subject_id(value):
    """Normalise l'identifiant d'un participant."""
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip()

    if not normalized:
        return pd.NA

    try:
        numeric = float(normalized)

        if numeric.is_integer():
            return str(int(numeric))

    except (TypeError, ValueError):
        pass

    return normalized


def normalize_text(value):
    """Normalise une chaîne pour les comparaisons techniques."""
    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = normalized.rstrip("/")

    return normalized or None


def normalize_task_type(value):
    """Normalise le type de tâche vers MP, MT, AC ou DA."""
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().upper()

    if normalized in EXPECTED_TASK_TYPES:
        return normalized

    return pd.NA


def normalize_response(value):
    """Normalise une réponse binaire vers Yes ou No."""
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"

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

    return pd.NA


def normalize_binary(value):
    """Normalise une valeur binaire vers 0 ou 1."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        numeric = float(value)

        if numeric in {0.0, 1.0}:
            return int(numeric)

    normalized = str(value).strip().lower()

    true_values = {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "oui",
        "correct",
    }

    false_values = {
        "0",
        "0.0",
        "false",
        "no",
        "n",
        "non",
        "incorrect",
    }

    if normalized in true_values:
        return 1

    if normalized in false_values:
        return 0

    return np.nan


def normalize_validity(value):
    """Transforme Valid en 1 et Invalid en 0."""
    if pd.isna(value):
        return np.nan

    normalized = str(value).strip().lower()

    if normalized in {
        "valid",
        "true",
        "1",
        "1.0",
        "yes",
    }:
        return 1

    if normalized in {
        "invalid",
        "false",
        "0",
        "0.0",
        "no",
    }:
        return 0

    return np.nan


def normalize_condition(value):
    """Normalise la condition vers Standard ou Neutral."""
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().lower()

    if normalized == "standard":
        return "Standard"

    if normalized in {
        "neutral",
        "neutre",
    }:
        return "Neutral"

    return pd.NA


def split_formal_task(value):
    """Sépare une tâche formelle en deux prémisses."""
    if pd.isna(value):
        return pd.NA, pd.NA

    parts = [
        part.strip()
        for part in str(value).split("/")
        if part.strip()
    ]

    if len(parts) != 2:
        return pd.NA, pd.NA

    return parts[0], parts[1]


def infer_task_type_from_premises(
    premise_1,
    premise_2,
):
    """Déduit MP, MT, AC ou DA depuis les prémisses."""
    normalized_pair = (
        normalize_text(premise_1),
        normalize_text(premise_2),
    )

    return TASK_MAPPING.get(
        normalized_pair,
        pd.NA,
    )


def build_normalized_formal_task(
    premise_1,
    premise_2,
):
    """Construit une représentation normalisée des prémisses."""
    normalized_1 = normalize_text(premise_1)
    normalized_2 = normalize_text(premise_2)

    if normalized_1 is None or normalized_2 is None:
        return pd.NA

    return f"{normalized_1}/{normalized_2}"


# ======================================================================
# ENTROPIE
# ======================================================================

def binary_entropy_from_probability(probability):
    """Calcule l'entropie binaire de Shannon."""
    if pd.isna(probability):
        return np.nan

    probability = float(probability)

    if probability < 0 or probability > 1:
        return np.nan

    if (
        np.isclose(probability, 0.0)
        or np.isclose(probability, 1.0)
    ):
        return 0.0

    complement = 1.0 - probability

    if ENTROPY_LOG_BASE == 2:
        log_function = math.log2

    else:
        def log_function(value):
            return math.log(
                value,
                ENTROPY_LOG_BASE,
            )

    return float(
        -(
            probability
            * log_function(probability)
            + complement
            * log_function(complement)
        )
    )


def compute_item_entropy(dataframe):
    """Calcule uniquement l'entropie nécessaire pour chaque item."""
    item_rows = []

    for item_id, item_data in dataframe.groupby(
        "item_id",
        sort=True,
    ):
        responses = (
            item_data["response_normalized"]
            .dropna()
        )

        yes_count = int(
            (responses == "Yes").sum()
        )

        no_count = int(
            (responses == "No").sum()
        )

        response_count = (
            yes_count
            + no_count
        )

        if (
            response_count
            >= MIN_RESPONSES_FOR_ENTROPY
        ):
            yes_rate = (
                yes_count
                / response_count
            )

            entropy = (
                binary_entropy_from_probability(
                    yes_rate
                )
            )

        else:
            entropy = np.nan

        item_rows.append({
            "item_id": item_id,
            "item_entropy": entropy,
            "item_n_responses": response_count,
            "item_n_subjects": int(
                item_data[
                    "subject_id"
                ].nunique()
            ),
        })

    return pd.DataFrame(item_rows)


# ======================================================================
# CHARGEMENT DES DONNÉES EXPÉRIMENTALES
# ======================================================================

def load_experiment_data():
    """Charge et normalise le fichier expérimental."""
    audit_section(
        "CHARGEMENT DU FICHIER EXPÉRIMENTAL"
    )

    if not os.path.isfile(EXPERIMENT_FILE):
        raise FileNotFoundError(
            "Fichier expérimental introuvable : "
            f"{EXPERIMENT_FILE}"
        )

    dataframe = pd.read_csv(
        EXPERIMENT_FILE
    )

    audit_print(
        "Fichier :",
        EXPERIMENT_FILE,
    )

    audit_print(
        "Nombre de lignes brutes :",
        len(dataframe),
    )

    required_columns = {
        "id",
        "sequence",
        "task",
        "response",
        "confidence",
        "is_correct",
        "task_type",
        "condition",
        "validity",
        "total_qnum",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Colonnes expérimentales manquantes : "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.rename(
        columns={
            "id": "subject_id",
            "task": "task_formal",
            "total_qnum": "item_id",
        }
    ).copy()

    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .apply(normalize_subject_id)
        .astype("string")
    )

    dataframe["sequence"] = pd.to_numeric(
        dataframe["sequence"],
        errors="coerce",
    )

    dataframe["item_id"] = pd.to_numeric(
        dataframe["item_id"],
        errors="coerce",
    )

    dataframe["confidence"] = pd.to_numeric(
        dataframe["confidence"],
        errors="coerce",
    )

    dataframe["is_correct"] = (
        dataframe["is_correct"]
        .apply(normalize_binary)
    )

    dataframe["validity_binary"] = (
        dataframe["validity"]
        .apply(normalize_validity)
    )

    dataframe["response_normalized"] = (
        dataframe["response"]
        .apply(normalize_response)
        .astype("string")
    )

    dataframe["response_binary"] = (
        dataframe["response_normalized"]
        .map({
            "No": 0,
            "Yes": 1,
        })
        .astype("Float64")
    )

    dataframe["task_type"] = (
        dataframe["task_type"]
        .apply(normalize_task_type)
        .astype("string")
    )

    dataframe["condition"] = (
        dataframe["condition"]
        .apply(normalize_condition)
        .astype("string")
    )

    extracted_premises = (
        dataframe["task_formal"]
        .apply(split_formal_task)
    )

    dataframe["experiment_premise_1"] = (
        extracted_premises
        .apply(lambda value: value[0])
    )

    dataframe["experiment_premise_2"] = (
        extracted_premises
        .apply(lambda value: value[1])
    )

    dataframe["task_formal_normalized"] = (
        dataframe.apply(
            lambda row:
                build_normalized_formal_task(
                    row["experiment_premise_1"],
                    row["experiment_premise_2"],
                ),
            axis=1,
        )
    )

    dataframe["task_type_inferred"] = (
        dataframe.apply(
            lambda row:
                infer_task_type_from_premises(
                    row["experiment_premise_1"],
                    row["experiment_premise_2"],
                ),
            axis=1,
        )
    )

    task_type_mismatch = (
        dataframe["task_type_inferred"].notna()
        & dataframe["task_type"].notna()
        & (
            dataframe["task_type_inferred"].astype(str)
            != dataframe["task_type"].astype(str)
        )
    )

    mismatch_count = int(
        task_type_mismatch.sum()
    )

    audit_print(
        "Incohérences task_type/prémisses :",
        mismatch_count,
    )

    if mismatch_count > 0:
        errors = dataframe.loc[
            task_type_mismatch,
                ["subject_id",
                "sequence",
                "item_id",
                "task_formal",
                "task_type",
                "task_type_inferred",
            ],
        ]

        errors.to_csv(
            ITEM_CONSISTENCY_ERRORS_FILE,
            index=False,
        )

        raise ValueError(
            "Certaines tâches possèdent un type incompatible "
            "avec leurs prémisses."
        )

    invalid_confidence = (
        dataframe["confidence"].notna()
        & (
            (dataframe["confidence"] < 0)
            | (dataframe["confidence"] > 100)
        )
    )

    audit_print(
        "Confiances hors de [0, 100] :",
        int(invalid_confidence.sum()),
    )

    if invalid_confidence.any():
        raise ValueError(
            "Certaines valeurs de confiance sont hors de [0, 100]."
        )

    before_drop = len(dataframe)

    dataframe = dataframe.dropna(
        subset=[
            "subject_id",
            "sequence",
            "item_id",
            "task_type",
        ]
    ).copy()

    audit_print(
        "Lignes supprimées pour identifiant manquant :",
        before_drop - len(dataframe),
    )

    dataframe["sequence"] = (
        dataframe["sequence"]
        .astype(int)
    )

    dataframe["item_id"] = (
        dataframe["item_id"]
        .astype(int)
    )

    duplicate_mask = dataframe.duplicated(
        subset=[
            "subject_id",
            "sequence",
        ],
        keep=False,
    )

    audit_print(
        "Doublons subject_id + sequence :",
        int(duplicate_mask.sum()),
    )

    if duplicate_mask.any():
        duplicate_rows = dataframe.loc[
            duplicate_mask,
            [
                "subject_id",
                "sequence",
                "item_id",
                "task_type",
            ],
        ]

        duplicate_rows.to_csv(
            ITEM_CONSISTENCY_ERRORS_FILE,
            index=False,
        )

        raise ValueError(
            "La clé subject_id + sequence n'est pas unique."
        )

    audit_print(
        "Participants :",
        dataframe["subject_id"].nunique(),
    )

    audit_print(
        "Items :",
        dataframe["item_id"].nunique(),
    )

    return dataframe


# ======================================================================
# FICHIER MREASONER
# ======================================================================

def load_model_count_data():
    """Charge et contrôle le fichier MReasoner."""
    audit_section(
        "CHARGEMENT DU FICHIER MREASONER"
    )

    if not os.path.isfile(MODEL_COUNT_FILE):
        raise FileNotFoundError(
            "Fichier MReasoner introuvable : "
            f"{MODEL_COUNT_FILE}"
        )

    dataframe = pd.read_csv(
        MODEL_COUNT_FILE
    )

    audit_print(
        "Fichier :",
        MODEL_COUNT_FILE,
    )

    audit_print(
        "Nombre de lignes :",
        len(dataframe),
    )

    required_columns = {
        "subject_id",
        "premise_1",
        "premise_2",
        "number_models_generated",
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Colonnes MReasoner manquantes : "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .apply(normalize_subject_id)
        .astype("string")
    )

    numeric_columns = [
        "number_models_generated",
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["model_task_type"] = (
        dataframe.apply(
            lambda row:
                infer_task_type_from_premises(
                    row["premise_1"],
                    row["premise_2"],
                ),
            axis=1,
        )
        .astype("string")
    )

    dataframe[
        "model_task_formal_normalized"
    ] = dataframe.apply(
        lambda row:
            build_normalized_formal_task(
                row["premise_1"],
                row["premise_2"],
            ),
        axis=1,
    )

    unknown_tasks = dataframe[
        "model_task_type"
    ].isna()

    audit_print(
        "Types de tâche MReasoner inconnus :",
        int(unknown_tasks.sum()),
    )

    if unknown_tasks.any():
        errors = dataframe.loc[
            unknown_tasks,
            [
                "subject_id",
                "premise_1",
                "premise_2",
            ],
        ]

        errors.to_csv(
            MODEL_STRUCTURE_ERRORS_FILE,
            index=False,
        )

        raise ValueError(
            "Certaines tâches MReasoner ne peuvent pas être reconnues."
        )

    duplicate_mask = dataframe.duplicated(
        subset=[
            "subject_id",
            "model_task_type",
        ],
        keep=False,
    )

    audit_print(
        "Doublons participant × tâche MReasoner :",
        int(duplicate_mask.sum()),
    )

    if duplicate_mask.any():
        errors = dataframe.loc[
            duplicate_mask
        ]

        errors.to_csv(
            MODEL_STRUCTURE_ERRORS_FILE,
            index=False,
        )

        raise ValueError(
            "Plusieurs lignes MReasoner existent pour une même "
            "combinaison participant × type de tâche."
        )

    model_structure = (
        dataframe
        .groupby("subject_id") ["model_task_type"]
        .agg(
            number_of_task_types="nunique",
            task_types=lambda values:
                ",".join(sorted(values.dropna().unique())),
        )
        .reset_index()
    )

    invalid_profiles = model_structure.loc[
        (
            model_structure[
                "number_of_task_types"
            ] != 4
        )
        | (
            model_structure[
                "task_types"
            ] != "AC,DA,MP,MT"
        )
    ]

    audit_print(
        "Participants MReasoner :",
        dataframe["subject_id"].nunique(),
    )

    audit_print(
        "Participants sans les quatre types de tâche :",
        len(invalid_profiles),
    )

    if not invalid_profiles.empty:
        invalid_profiles.to_csv(
            MODEL_STRUCTURE_ERRORS_FILE,
            index=False,
        )

        if STRICT_VALIDATION:
            raise ValueError(
                "Certains participants MReasoner ne possèdent pas "
                "exactement les quatre types de tâche."
            )

    if "n_samples" in dataframe.columns:
        audit_print(
            "Nombre de simulations MReasoner :",
            sorted(
                dataframe["n_samples"]
                .dropna()
                .unique()
                .tolist()
            ),
        )

    for parameter in [
        "n_parameter_sets_used",
        "epsilon",
        "lambda",
        "omega",
        "sigma",
    ]:
        if parameter in dataframe.columns:
            unique_values = (
                dataframe[parameter]
                .dropna()
                .unique()
                .tolist()
            )

            audit_print(
                f"Paramètre MReasoner {parameter} :",
                unique_values,
            )

    return dataframe


# ======================================================================
# CONTRÔLE DE LA COHÉRENCE DES ITEMS
# ======================================================================

def audit_item_consistency(dataframe):
    """Vérifie qu'un item possède une définition unique."""
    audit_section(
        "COHÉRENCE DES ITEMS"
    )

    consistency_columns = [
        "condition",
        "task_type",
        "task_formal",
        "validity_binary",
    ]

    item_consistency = (
        dataframe
        .groupby("item_id")[consistency_columns]
        .nunique(dropna=True)
        .reset_index()
    )

    inconsistent_mask = (
        item_consistency[
            consistency_columns
        ]
        .gt(1)
        .any(axis=1)
    )

    inconsistent_items = item_consistency.loc[
        inconsistent_mask
    ]

    audit_print(
        "Items distincts :",
        len(item_consistency),
    )

    audit_print(
        "Items incohérents :",
        len(inconsistent_items),
    )

    if not inconsistent_items.empty:
        inconsistent_items.to_csv(
            ITEM_CONSISTENCY_ERRORS_FILE,
            index=False,
        )

        if STRICT_VALIDATION:
            raise ValueError(
                "Certains items possèdent plusieurs définitions."
            )


# ======================================================================
# VARIABLES ANALYTIQUES
# ======================================================================

def add_subject_accuracy(dataframe):
    """Ajoute la précision globale du participant."""
    audit_section(
        "PRÉCISION PAR PARTICIPANT"
    )

    subject_accuracy = (
        dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            subject_accuracy=(
                "is_correct",
                "mean",
            ),
            subject_n_trials=(
                "sequence",
                "size",
            ),
            subject_condition_count=(
                "condition",
                "nunique",
            ),
        )
    )

    invalid_conditions = subject_accuracy.loc[
        subject_accuracy[
            "subject_condition_count"
        ] != 1
    ]

    if not invalid_conditions.empty:
        invalid_conditions.to_csv(
            ITEM_CONSISTENCY_ERRORS_FILE,
            index=False,
        )

        raise ValueError(
            "Certains participants apparaissent dans plusieurs "
            "conditions."
        )

    audit_print(
        "Précision moyenne entre participants :",
        round(
            subject_accuracy[
                "subject_accuracy"
            ].mean(),
            6,
        ),
    )

    audit_print(
        "Essais par participant — minimum :",
        int(
            subject_accuracy[
                "subject_n_trials"
            ].min()
        ),
    )

    audit_print(
        "Essais par participant — maximum :",
        int(
            subject_accuracy[
                "subject_n_trials"
            ].max()
        ),
    )

    return dataframe.merge(
        subject_accuracy[[
            "subject_id",
            "subject_accuracy",
        ]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )


def add_item_entropy(dataframe):
    """Calcule et fusionne l'entropie des items."""
    audit_section(
        "ENTROPIE PAR ITEM"
    )

    item_summary = compute_item_entropy(
        dataframe
    )

    audit_print(
        "Items avec une entropie calculable :",
        int(
            item_summary[
                "item_entropy"
            ].notna()
            .sum()
        ),
    )

    audit_print(
        "Entropie minimale :",
        round(
            item_summary[
                "item_entropy"
            ].min(),
            6,
        ),
    )

    audit_print(
        "Entropie médiane :",
        round(
            item_summary[
                "item_entropy"
            ].median(),
            6,
        ),
    )

    audit_print(
        "Entropie maximale :",
        round(
            item_summary[
                "item_entropy"
            ].max(),
            6,
        ),
    )

    return dataframe.merge(
        item_summary[[
            "item_id",
            "item_entropy",
        ]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )


def merge_model_counts(
    experiment_data,
    model_data,
):
    """Fusionne les essais expérimentaux avec MReasoner."""
    audit_section(
        "FUSION AVEC MREASONER"
    )

    model_subset = model_data[[
        "subject_id",
        "model_task_type",
        "model_task_formal_normalized",
        "number_models_generated",
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",
    ]].copy()

    model_subset = model_subset.rename(
        columns={
            "model_task_type":
                "task_type",
        }
    )

    merged = experiment_data.merge(
        model_subset,
        on=[
            "subject_id",
            "task_type",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched_mask = (
        merged["_merge"] == "both"
    )

    premise_match = (
        merged["task_formal_normalized"]
        == merged[
            "model_task_formal_normalized"
        ]
    )

    missing_model_mask = (
        ~matched_mask
        | merged[
            "number_models_generated"
        ].isna()
    )

    premise_mismatch_mask = (
        matched_mask
        & ~premise_match.fillna(False)
    )

    error_mask = (
        missing_model_mask
        | premise_mismatch_mask
    )

    audit_print(
        "Essais correctement appariés :",
        int((~error_mask).sum()),
    )

    audit_print(
        "Essais sans comptage MReasoner :",
        int(missing_model_mask.sum()),
    )

    audit_print(
        "Essais avec prémisses incompatibles :",
        int(premise_mismatch_mask.sum()),
    )

    if error_mask.any():
        error_rows = merged.loc[
            error_mask,
            [
                "subject_id",
                "sequence",
                "item_id",
                "task_type",
                "task_formal",
                "number_models_generated",
                "_merge",
            ],
        ]

        error_rows.to_csv(
            MODEL_MERGE_ERRORS_FILE,
            index=False,
        )

        if STRICT_VALIDATION:
            raise ValueError(
                "La fusion avec MReasoner contient des erreurs. "
                f"Consultez {MODEL_MERGE_ERRORS_FILE}."
            )

    merged = merged.drop(
        columns=[
            "_merge",
            "model_task_formal_normalized",
        ]
    )

    return merged


def add_model_decomposition(dataframe):
    """Ajoute les composantes inter- et intra-individuelles."""
    audit_section(
        "DÉCOMPOSITION DU NOMBRE DE MODÈLES"
    )

    subject_mean_models = (
        dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            subject_mean_models=(
                "number_models_generated",
                "mean",
            ),
        )
    )

    dataframe = dataframe.merge(
        subject_mean_models,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    dataframe[
        "models_within_subject"
    ] = (
        dataframe[
            "number_models_generated"
        ]
        - dataframe[
            "subject_mean_models"
        ]
    )

    task_values_per_subject = (
        dataframe[[
            "subject_id",
            "task_type",
            "number_models_generated",
        ]]
        .drop_duplicates(
            subset=[
                "subject_id",
                "task_type",]
        )
        .groupby("subject_id")[
            "number_models_generated"
        ]
        .nunique()
    )

    no_variation_count = int(
        (task_values_per_subject <= 1).sum()
    )

    audit_print(
        "Moyenne globale du nombre de modèles :",
        round(
            dataframe[
                "number_models_generated"
            ].mean(),
            6,
        ),
    )

    audit_print(
        "Participants sans variation entre tâches :",
        no_variation_count,
    )

    return dataframe


def add_analysis_variables(dataframe):
    """Ajoute les indicateurs utilisés par les scripts de modèle."""
    dataframe["is_ceiling"] = (
        dataframe["confidence"] == 100
    ).astype(int)

    primary_columns = [
        "confidence",
        "condition",
        "sequence",
        "subject_id",
        "item_id",
        "task_type",
        "validity_binary",
        "subject_accuracy",
        "item_entropy",
        "number_models_generated",
        "subject_mean_models",
        "models_within_subject",
    ]

    dataframe["analysis_complete"] = (
        dataframe[
            primary_columns
        ]
        .notna()
        .all(axis=1)
    )

    return dataframe


# ======================================================================
# AUDIT FINAL
# ======================================================================

def audit_final_data(dataframe):
    """Produit un audit concis du dataset final."""
    audit_section(
        "AUDIT FINAL"
    )

    audit_print(
        "Nombre de lignes :",
        len(dataframe),
    )

    audit_print(
        "Nombre de participants :",
        dataframe["subject_id"].nunique(),
    )

    audit_print(
        "Nombre d'items :",
        dataframe["item_id"].nunique(),
    )

    audit_print(
        "Nombre de lignes complètes :",
        int(
            dataframe[
                "analysis_complete"
            ].sum()
        ),
    )

    audit_print(
        "Nombre de lignes incomplètes :",
        int(
            (
                ~dataframe[
                    "analysis_complete"
                ]
            ).sum()
        ),
    )

    audit_print(
        "Confiance moyenne :",
        round(
            dataframe[
                "confidence"
            ].mean(),
            6,
        ),
    )

    audit_print(
        "Confiances égales à 100 :",
        int(
            dataframe[
                "is_ceiling"
            ].sum()
        ),
        (
            f"({100 * dataframe['is_ceiling'].mean():.3f} %)"
        ),
    )

    condition_counts = (
        dataframe[[
            "subject_id",
            "condition",
        ]]
        .drop_duplicates()
        ["condition"]
        .value_counts()
    )

    audit_print(
        "Participants par condition :"
    )

    for condition, count in condition_counts.items():
        audit_print(
            f"  {condition} :",
            int(count),
        )

    task_counts = (
        dataframe[
            "task_type"
        ]
        .value_counts()
    )

    audit_print(
        "Essais par type de tâche :"
    )

    for task_type, count in task_counts.items():
        audit_print(
            f"  {task_type} :",
            int(count),
        )

    validity_table = pd.crosstab(
        dataframe["task_type"],
        dataframe["validity_binary"],
    )

    audit_print(
        "Relation type de tâche × validité :"
    )

    audit_print(
        validity_table.to_string()
    )


# ======================================================================
# SÉLECTION DES COLONNES FINALES
# ======================================================================

def select_final_columns(dataframe):
    """Conserve uniquement les colonnes nécessaires aux modèles."""
    final_columns = [
        # Identifiants
        "subject_id",
        "sequence",
        "item_id",

        # Variable dépendante
        "confidence",

        # Réponse et exactitude
        "response_normalized",
        "response_binary",
        "is_correct",

        # Structure expérimentale
        "condition",
        "task_type",
        "validity_binary",

        # Prédicteurs
        "subject_accuracy",
        "item_entropy",
        "number_models_generated",
        "subject_mean_models",
        "models_within_subject",

        # Informations sur les simulations
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",

        # Analyses de plafond et complétude
        "is_ceiling",
        "analysis_complete",
    ]

    missing_columns = [
        column
        for column in final_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes finales absentes : "
            f"{missing_columns}"
        )

    return dataframe[
        final_columns
    ].copy()


# ======================================================================
# NETTOYAGE DES ANCIENS FICHIERS D'ERREUR
# ======================================================================

def remove_old_error_files():
    """
    Supprime les fichiers d'erreur d'une exécution précédente.

    Ainsi, un ancien fichier d'erreur ne peut pas être confondu avec
    une erreur de l'exécution actuelle.
    """
    for path in [
        MODEL_MERGE_ERRORS_FILE,
        ITEM_CONSISTENCY_ERRORS_FILE,
        MODEL_STRUCTURE_ERRORS_FILE,
    ]:
        if os.path.isfile(path):
            os.remove(path)


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print("CONSTRUCTION DU DATASET ANALYTIQUE E1 — VERSION MINIMALE")
    print("=" * 80)

    remove_old_error_files()

    try:
        # 1. Chargement
        experiment_data = (
            load_experiment_data()
        )

        model_data = (
            load_model_count_data()
        )

        # 2. Contrôle des items
        audit_item_consistency(
            experiment_data
        )

        # 3. Variables humaines
        experiment_data = (
            add_subject_accuracy(
                experiment_data
            )
        )

        experiment_data = (
            add_item_entropy(
                experiment_data
            )
        )

        # 4. Fusion avec MReasoner
        analysis_data = merge_model_counts(
            experiment_data=experiment_data,
            model_data=model_data,
        )

        # 5. Décomposition MReasoner
        analysis_data = (
            add_model_decomposition(
                analysis_data
            )
        )

        # 6. Variables finales
        analysis_data = (
            add_analysis_variables(
                analysis_data
            )
        )

        # 7. Tri
        analysis_data = (
            analysis_data
            .sort_values(
                by=[
                    "subject_id",
                    "sequence",
                ]
            )
            .reset_index(drop=True)
        )

        # 8. Audit avant réduction des colonnes
        audit_final_data(
            analysis_data
        )

        # 9. Sélection minimale
        final_data = select_final_columns(
            analysis_data
        )

        # 10. Validation finale stricte
        incomplete_count = int(
            (
                ~final_data[
                    "analysis_complete"
                ]
            ).sum()
        )

        if (
            incomplete_count > 0
            and STRICT_VALIDATION
        ):
            raise ValueError(
                f"{incomplete_count} ligne(s) sont incomplètes "
                "pour les modèles principaux."
            )

        # 11. Sauvegarde
        final_data.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        audit_section(
            "RÉSUMÉ"
        )

        audit_print(
            "Dataset créé :",
            OUTPUT_FILE,
        )

        audit_print(
            "Nombre de lignes :",
            len(final_data),
        )

        audit_print(
            "Nombre de colonnes :",
            len(final_data.columns),
        )

        audit_print(
            "Colonnes conservées :",
            list(final_data.columns),
        )

        audit_print(
            "Fichier d'audit :",
            AUDIT_REPORT_FILE,
        )

        audit_print(
            "Aucun fichier de diagnostic détaillé n'est créé "
            "lorsque les validations réussissent."
        )

        save_audit_report()

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        TypeError,
        pd.errors.ParserError,
        pd.errors.MergeError,
    ) as error:
        audit_section(
            "ERREUR"
        )

        audit_print(
            type(error).__name__,
            ":",
            error,
        )

        save_audit_report()

        sys.exit(1)


if __name__ == "__main__":
    main()