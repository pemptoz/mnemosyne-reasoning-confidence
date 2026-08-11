"""
build_dataset_analysis_E1.py

Construit le fichier analytique E1 utilisé pour prédire la confiance.

Fichiers d'entrée :
    dataset_ccobra_E1.csv
    mental_models_count_E1_n20.csv

Fichier principal produit :
    dataset_analysis_E1_n20.csv

Fichiers de diagnostic produits :
    analysis_E1_outputs/item_entropy_summary_E1.csv
    analysis_E1_outputs/subject_summary_E1.csv
    analysis_E1_outputs/model_merge_diagnostic_E1.csv
    analysis_E1_outputs/predictor_correlations_E1.csv
    analysis_E1_outputs/data_audit_E1.txt

Structure retenue :

    dataset_ccobra_E1.csv :
        une ligne = participant × essai × item

    mental_models_count_E1_n20.csv :
        une ligne = participant × type formel de syllogisme

La fusion du nombre de modèles est donc effectuée sur :

    subject_id + task_type

et non sur :

    subject_id + sequence

Variables construites :

    subject_accuracy
        Précision globale du participant.

    item_entropy
        Entropie binaire de Shannon des réponses Yes/No pour chaque
        total_qnum.

    item_accuracy
        Précision collective pour chaque item.

    subject_mean_models
        Nombre moyen de modèles du participant, tous types de tâches
        confondus.

    models_within_subject
        Écart entre le nombre de modèles de la ligne et la moyenne
        personnelle du participant.

    analysis_complete
        Indique si toutes les variables principales nécessaires au
        modèle sont disponibles.

Aucune régression et aucun modèle mixte ne sont ajustés ici.
Ce script prépare et audite uniquement les données.
"""

import json
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EXPERIMENT_FILE = os.path.join(
    BASE_DIR,
    "../dataset_ccobra_E1.csv",
)

MODEL_COUNT_FILE = os.path.join(
    BASE_DIR,
    "mental_models_count_E1_n20.csv",
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset_analysis_E1_n20.csv",
)

OUTPUT_DIRECTORY = os.path.join(
    BASE_DIR,
    "analysis_E1_outputs",
)

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# Fichiers de diagnostic
# ----------------------------------------------------------------------

ITEM_ENTROPY_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "item_entropy_summary_E1.csv",
)

SUBJECT_SUMMARY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "subject_summary_E1.csv",
)

MODEL_MERGE_DIAGNOSTIC_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "model_merge_diagnostic_E1.csv",
)

PREDICTOR_CORRELATIONS_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "predictor_correlations_E1.csv",
)

AUDIT_REPORT_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "data_audit_E1.txt",
)

ITEM_CONSISTENCY_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "item_consistency_E1.csv",
)

MODEL_STRUCTURE_FILE = os.path.join(
    OUTPUT_DIRECTORY,
    "model_count_structure_E1.csv",
)


# ----------------------------------------------------------------------
# Options méthodologiques
# ----------------------------------------------------------------------

# Base 2 : l'entropie binaire varie de 0 à 1 bit.
ENTROPY_LOG_BASE = 2

# Nombre minimum de réponses humaines nécessaires pour calculer
# l'entropie d'un item.
MIN_RESPONSES_FOR_ENTROPY = 2

# Le fichier final conserve toutes les lignes humaines, même lorsqu'un
# comptage mReasoner est absent. La colonne analysis_complete permettra
# d'identifier les lignes prêtes pour le modèle.
KEEP_INCOMPLETE_ROWS = True

# Si True, le programme s'arrête lorsqu'une incompatibilité majeure
# de structure est détectée.
STRICT_VALIDATION = True


# ======================================================================
# CORRESPONDANCE DES TÂCHES FORMELLES
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


# ======================================================================
# JOURNAL D'AUDIT
# ======================================================================

AUDIT_LINES = []


def audit_print(*values):
    """
    Affiche une information dans le terminal et l'ajoute au rapport
    d'audit.
    """
    text = " ".join(
        str(value)
        for value in values
    )

    print(text)
    AUDIT_LINES.append(text)


def audit_section(title):
    """
    Ajoute un titre de section dans le terminal et dans le rapport.
    """
    separator = "=" * 80

    audit_print("")
    audit_print(separator)
    audit_print(title)
    audit_print(separator)


def save_audit_report():
    """
    Sauvegarde le rapport d'audit dans un fichier texte.
    """
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
# NORMALISATION GÉNÉRALE
# ======================================================================

def normalize_subject_id(value):
    """
    Normalise un identifiant de participant.

    Exemples :
        63873     -> "63873"
        63873.0   -> "63873"
        "63873"   -> "63873"
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip()

    if not normalized:
        return pd.NA

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


def normalize_text(value):
    """
    Normalise une chaîne pour les comparaisons techniques.

    Cette fonction :
        - convertit en minuscules ;
        - supprime les espaces en début et fin ;
        - remplace les suites d'espaces par un espace unique ;
        - retire un slash final éventuel.
    """
    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = normalized.rstrip(
        "/"
    )

    return normalized or None


def normalize_task_type(value):
    """
    Normalise le type de syllogisme vers MP, MT, AC ou DA.
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().upper()

    if normalized in {
        "MP",
        "MT",
        "AC",
        "DA",
    }:
        return normalized

    return pd.NA


def normalize_response(value):
    """
    Normalise une réponse binaire vers Yes ou No.
    """
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (bool, np.bool_)):
        return (
            "Yes"
            if bool(value)
            else "No"
        )

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
    """
    Normalise une valeur binaire vers 0 ou 1.
    """
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

        if numeric == 1:
            return 1

        if numeric == 0:
            return 0

    normalized = str(value).strip().lower()

    true_values = {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "oui",
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
        "non",
        "incorrect",
        "invalid",
        "unbelievable",
        "no-conflict",
        "no conflict",
        "non-conflict",
    }

    if normalized in true_values:
        return 1

    if normalized in false_values:
        return 0

    return np.nan


def normalize_validity(value):
    """
    Normalise la validité logique :

        Valid   -> 1
        Invalid -> 0
    """
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
    """
    Normalise la condition vers Standard ou Neutral.
    """
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

    return str(value).strip()


def split_formal_task(value):
    """
    Sépare une tâche formelle en deux prémisses.

    Exemple :
        All B are C/No A are C

    retourne :
        ("All B are C", "No A are C")
    """
    if pd.isna(value):
        return (
            pd.NA,
            pd.NA,
        )

    raw_value = str(value).strip()

    parts = [
        part.strip()
        for part in raw_value.split("/")
        if part.strip()
    ]

    if len(parts) != 2:
        return (
            pd.NA,
            pd.NA,
        )

    return tuple(parts)


def infer_task_type_from_premises(
    premise_1,
    premise_2,
):
    """
    Déduit MP, MT, AC ou DA depuis les prémisses formelles.
    """
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
    """
    Construit une version normalisée de la tâche formelle.
    """
    normalized_1 = normalize_text(
        premise_1
    )

    normalized_2 = normalize_text(
        premise_2
    )

    if (
        normalized_1 is None
        or normalized_2 is None
    ):
        return pd.NA

    return (
        f"{normalized_1}/"
        f"{normalized_2}"
    )


# ======================================================================
# ENTROPIE
# ======================================================================

def binary_entropy_from_probability(probability):
    """
    Calcule l'entropie binaire de Shannon.

    H(p) = -p log2(p) - (1-p) log2(1-p)

    La valeur est comprise entre :
        0 : accord total ;
        1 : répartition 50/50.

    La convention 0 * log(0) = 0 est appliquée.
    """
    if pd.isna(probability):
        return np.nan

    probability = float(
        probability
    )

    if (
        probability < 0
        or probability > 1
    ):
        return np.nan

    if (
        np.isclose(probability, 0.0)
        or np.isclose(probability, 1.0)
    ):
        return 0.0

    complement = (
        1.0 - probability
    )

    if ENTROPY_LOG_BASE == 2:
        log_function = math.log2

    else:
        def log_function(value):
            return math.log(
                value,
                ENTROPY_LOG_BASE,
            )

    entropy = -(
        probability
        * log_function(probability)
        + complement
        * log_function(complement)
    )

    return float(entropy)


def compute_item_statistics(dataframe):
    """
    Calcule les statistiques collectives par item_id.

    L'entropie est calculée à partir de la distribution Yes/No.
    """
    item_rows = []

    grouped_items = dataframe.groupby(
        "item_id",
        dropna=False,
        sort=True,
    )

    for item_id, item_data in grouped_items:
        responses = (
            item_data["response_normalized"]
            .dropna()
        )

        yes_count = int(
            (
                responses == "Yes"
            ).sum()
        )

        no_count = int(
            (
                responses == "No"
            ).sum()
        )

        response_count = (
            yes_count + no_count
        )

        if response_count > 0:
            yes_rate = (
                yes_count
                / response_count
            )

            no_rate = (
                no_count
                / response_count
            )

        else:
            yes_rate = np.nan
            no_rate = np.nan

        if (
            response_count
            >= MIN_RESPONSES_FOR_ENTROPY
        ):
            item_entropy = (
                binary_entropy_from_probability(
                    yes_rate
                )
            )
        else:
            item_entropy = np.nan

        correct_values = (
            pd.to_numeric(
                item_data["is_correct"],
                errors="coerce",
            )
            .dropna()
        )

        item_accuracy = (
            float(correct_values.mean())
            if not correct_values.empty
            else np.nan
        )

        item_rows.append({
            "item_id":
                item_id,

            "item_n_rows":
                int(len(item_data)),

            "item_n_responses":
                int(response_count),

            "item_yes_count":
                yes_count,

            "item_no_count":
                no_count,

            "item_yes_rate":
                yes_rate,

            "item_no_rate":
                no_rate,

            "item_entropy":
                item_entropy,

            "item_accuracy":
                item_accuracy,

            "item_n_subjects":
                int(
                    item_data[
                        "subject_id"
                    ].nunique()
                ),
        })

    item_summary = pd.DataFrame(
        item_rows
    )

    return item_summary


# ======================================================================
# CHARGEMENT DES DONNÉES EXPÉRIMENTALES
# ======================================================================

def load_experiment_data():
    """
    Charge et normalise dataset_ccobra_E1.csv.
    """
    audit_section(
        "CHARGEMENT DU FICHIER EXPÉRIMENTAL"
    )

    if not os.path.isfile(
        EXPERIMENT_FILE
    ):
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

    audit_print(
        "Colonnes :",
        list(dataframe.columns),
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
        "believability",
        "conflict",
        "stimulus",
        "qnum",
        "total_qnum",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Colonnes absentes de "
            "dataset_ccobra_E1.csv : "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    # Renommages explicites.
    dataframe = dataframe.rename(
        columns={
            "id": "subject_id",
            "task": "task_formal",
            "total_qnum": "item_id",
        }
    )

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

    dataframe["qnum"] = pd.to_numeric(
        dataframe["qnum"],
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

    # Extraction des deux prémisses depuis task_formal.
    extracted_premises = (
        dataframe["task_formal"]
        .apply(split_formal_task)
    )

    dataframe["experiment_premise_1"] = (
        extracted_premises
        .apply(
            lambda value: value[0]
        )
    )

    dataframe["experiment_premise_2"] = (
        extracted_premises
        .apply(
            lambda value: value[1]
        )
    )

    dataframe[
        "task_formal_normalized"
    ] = dataframe.apply(
        lambda row:
            build_normalized_formal_task(
                row["experiment_premise_1"],
                row["experiment_premise_2"],
            ),
        axis=1,
    )

    dataframe[
        "task_type_inferred"
    ] = dataframe.apply(
        lambda row:
            infer_task_type_from_premises(
                row["experiment_premise_1"],
                row["experiment_premise_2"],
            ),
        axis=1,
    )

    # Contrôle du type déclaré contre le type déduit.
    task_type_mismatch = (
        dataframe[
            "task_type_inferred"
        ].notna()
        & dataframe[
            "task_type"
        ].notna()
        & (
            dataframe[
                "task_type_inferred"
            ].astype(str)
            != dataframe[
                "task_type"
            ].astype(str)
        )
    )

    mismatch_count = int(
        task_type_mismatch.sum()
    )

    audit_print(
        "Incohérences entre task_type et les prémisses :",
        mismatch_count,
    )

    if (
        mismatch_count > 0
        and STRICT_VALIDATION
    ):
        mismatch_preview = dataframe.loc[
            task_type_mismatch,
                [
                "subject_id",
                "sequence",
                "task_formal",
                "task_type",
                "task_type_inferred",
            ],
        ].head(20)

        raise ValueError(
            "Certaines tâches expérimentales ont un task_type "
            "incompatible avec leurs prémisses.\n"
            f"{mismatch_preview.to_string(index=False)}"
        )

    # Validation des bornes de confiance.
    invalid_confidence_mask = (
        dataframe["confidence"].notna()
        & (
            (
                dataframe["confidence"] < 0
            )
            | (
                dataframe["confidence"] > 100
            )
        )
    )

    invalid_confidence_count = int(
        invalid_confidence_mask.sum()
    )

    audit_print(
        "Confiances hors de [0, 100] :",
        invalid_confidence_count,
    )

    if (
        invalid_confidence_count > 0
        and STRICT_VALIDATION
    ):
        raise ValueError(
            "Certaines valeurs de confiance sont "
            "hors de l'intervalle [0, 100]."
        )

    # Validation des valeurs de correction.
    invalid_correctness = dataframe.loc[
        dataframe["is_correct"].notna()
        & ~dataframe["is_correct"].isin(
            [0, 1]
        ),
        "is_correct",
    ]

    if not invalid_correctness.empty:
        raise ValueError(
            "La colonne is_correct contient des valeurs "
            "différentes de 0 et 1 : "
            f"{sorted(invalid_correctness.unique())}"
        )

    # Les lignes sans identifiant principal ne peuvent pas être
    # correctement analysées.
    before_drop = len(
        dataframe
    )

    dataframe = (
        dataframe
        .dropna(
            subset=[
                "subject_id",
                "sequence",
                "item_id",
                "task_type",
            ]
        )
        .copy()
    )

    removed_rows = (
        before_drop - len(dataframe)
    )

    audit_print(
        "Lignes supprimées pour identifiant essentiel manquant :",
        removed_rows,
    )

    dataframe["sequence"] = (
        dataframe["sequence"]
        .astype(int)
    )

    dataframe["item_id"] = (
        dataframe["item_id"]
        .astype(int)
    )

    # Vérifie l'unicité participant × séquence.
    duplicate_key_mask = dataframe.duplicated(
        subset=[
            "subject_id",
            "sequence",
        ],
        keep=False,
    )

    duplicate_key_count = int(
        duplicate_key_mask.sum()
    )

    audit_print(
        "Lignes impliquées dans un doublon "
        "subject_id + sequence :",
        duplicate_key_count,
    )

    if (
        duplicate_key_count > 0
        and STRICT_VALIDATION
    ):
        duplicate_preview = dataframe.loc[
            duplicate_key_mask,
                [
                "subject_id",
                "sequence",
                "item_id",
                "task_type",
            ],
        ].head(20)

        raise ValueError(
            "La clé subject_id + sequence n'est pas unique "
            "dans le fichier expérimental.\n"
            f"{duplicate_preview.to_string(index=False)}"
        )

    audit_print(
        "Participants :",
        dataframe["subject_id"].nunique(),
    )

    audit_print(
        "Items distincts :",
        dataframe["item_id"].nunique(),
    )

    audit_print(
        "Types de tâches :",
        sorted(
            dataframe[
                "task_type"
            ]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    return dataframe


# ======================================================================
# CHARGEMENT DU FICHIER DE MODÈLES
# ======================================================================

def load_model_count_data():
    """
    Charge mental_models_count_E1_n20.csv.

    Le fichier semble contenir une ligne par :

        participant × type formel de syllogisme

    et non une ligne par séquence expérimentale.
    """
    audit_section(
        "CHARGEMENT DU FICHIER DE MODÈLES MENTAUX"
    )

    if not os.path.isfile(
        MODEL_COUNT_FILE
    ):
        raise FileNotFoundError(
            "Fichier de comptage des modèles introuvable : "
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
        "Nombre de lignes brutes :",
        len(dataframe),
    )

    audit_print(
        "Colonnes :",
        list(dataframe.columns),
    )

    required_columns = {
        "subject_id",
        "task",
        "premise_1",
        "premise_2",
        "number_models_generated",
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",
        "n_parameter_sets_used",
        "epsilon",
        "lambda",
        "omega",
        "sigma",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Colonnes absentes de "
            "mental_models_count_E1_n20.csv : "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    # Le nom task est ambigu dans ce fichier. On le conserve comme
    # index interne du type de tâche, sans l'utiliser comme séquence.
    dataframe = dataframe.rename(
        columns={
            "task": "model_task_index",
        }
    )

    dataframe["subject_id"] = (
        dataframe["subject_id"]
        .apply(normalize_subject_id)
        .astype("string")
    )

    dataframe[
        "model_task_index"
    ] = pd.to_numeric(
        dataframe["model_task_index"],
        errors="coerce",
    )

    numeric_columns = [
        "number_models_generated",
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",
        "n_parameter_sets_used",
        "epsilon",
        "lambda",
        "omega",
        "sigma",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe[
        "model_task_type"
    ] = dataframe.apply(
        lambda row:
            infer_task_type_from_premises(
                row["premise_1"],
                row["premise_2"],
            ),
        axis=1,
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

    unknown_task_mask = dataframe[
        "model_task_type"
    ].isna()

    unknown_task_count = int(
        unknown_task_mask.sum()
    )

    audit_print(
        "Lignes dont le type de tâche ne peut pas être déduit :",
        unknown_task_count,
    )

    if (
        unknown_task_count > 0
        and STRICT_VALIDATION
    ):
        unknown_preview = dataframe.loc[
            unknown_task_mask,
                [
                "subject_id",
                "model_task_index",
                "premise_1",
                "premise_2",
            ],
        ].head(20)

        raise ValueError(
            "Certaines prémisses du fichier de modèles ne "
            "correspondent à aucun type MP, MT, AC ou DA.\n"
            f"{unknown_preview.to_string(index=False)}"
        )

    dataframe[
        "model_task_type"
    ] = (
        dataframe[
            "model_task_type"
        ]
        .astype("string")
    )

    # Contrôle participant × type de tâche.
    duplicate_model_mask = dataframe.duplicated(
        subset=[
            "subject_id",
            "model_task_type",
        ],
        keep=False,
    )

    duplicate_model_count = int(
        duplicate_model_mask.sum()
    )

    audit_print(
        "Lignes impliquées dans un doublon "
        "subject_id + model_task_type :",
        duplicate_model_count,
    )

    if (
        duplicate_model_count > 0
        and STRICT_VALIDATION
    ):
        duplicate_preview = dataframe.loc[
            duplicate_model_mask,
            [
                "subject_id",
                "model_task_index",
                "model_task_type",
                "premise_1",
                "premise_2",
                "number_models_generated",
            ],
        ].head(30)

        raise ValueError(
            "Le fichier de modèles contient plusieurs lignes "
            "pour le même participant et le même type de tâche.\n"
            f"{duplicate_preview.to_string(index=False)}"
        )

    # Tableau de structure par participant.
    model_structure = (
        dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            number_of_model_rows=(
                "model_task_type",
                "size",
            ),
            number_of_task_types=(
                "model_task_type",
                "nunique",
            ),
            task_types=(
                "model_task_type",
                lambda values:
                    ",".join(
                        sorted(
                            set(
                                value
                                for value in values.dropna()
                            )
                        )
                    ),
            ),
            mean_number_models=(
                "number_models_generated",
                "mean",
            ),
        )
    )

    model_structure.to_csv(
        MODEL_STRUCTURE_FILE,
        index=False,
    )

    audit_print(
        "Participants dans le fichier de modèles :",
        dataframe["subject_id"].nunique(),
    )

    audit_print(
        "Nombre médian de lignes de modèles par participant :",
        float(
            model_structure[
                "number_of_model_rows"
            ].median()
        ),
    )

    audit_print(
        "Nombre maximal de lignes de modèles par participant :",
        int(
            model_structure[
                "number_of_model_rows"
            ].max()
        ),
    )

    incomplete_model_profiles = model_structure.loc[
        model_structure[
            "number_of_task_types"
        ] < 4
    ]

    audit_print(
        "Participants ayant moins de quatre types de tâches "
        "dans le fichier de modèles :",
        len(incomplete_model_profiles),
    )

    return dataframe


# ======================================================================
# CONTRÔLE DES ITEMS
# ======================================================================

def audit_item_consistency(dataframe):
    """
    Vérifie que chaque item_id possède une définition expérimentale
    cohérente.
    """
    audit_section(
        "COHÉRENCE DES ITEMS"
    )

    consistency_columns = [
        "condition",
        "task_type",
        "task_formal",
        "validity",
        "validity_binary",
        "believability",
        "conflict",
        "stimulus",
    ]

    aggregation = {}

    for column in consistency_columns:
        aggregation[
            f"nunique_{column}"
        ] = (
            column,
            lambda values:
                values.dropna().nunique()
        )

        aggregation[
            f"first_{column}"
        ] = (
            column,
            "first",
        )

    item_consistency = (
        dataframe
        .groupby(
            "item_id",
            as_index=False,
        )
        .agg(
            **aggregation
        )
    )

    inconsistency_columns = [
        f"nunique_{column}"
        for column in consistency_columns
    ]

    item_consistency[
        "is_consistent"
    ] = (
        item_consistency[
            inconsistency_columns
        ]
        .fillna(0)
        .le(1)
        .all(axis=1)
    )

    inconsistent_items = item_consistency.loc[
        ~item_consistency[
            "is_consistent"
        ]
    ]

    audit_print(
        "Items distincts :",
        len(item_consistency),
    )

    audit_print(
        "Items présentant au moins une incohérence :",
        len(inconsistent_items),
    )

    item_consistency.to_csv(
        ITEM_CONSISTENCY_FILE,
        index=False,
    )

    if (
        not inconsistent_items.empty
        and STRICT_VALIDATION
    ):
        raise ValueError(
            "Certains item_id correspondent à plusieurs "
            "définitions expérimentales. Consulte : "
            f"{ITEM_CONSISTENCY_FILE}"
        )

    return item_consistency


# ======================================================================
# PRÉCISION PAR PARTICIPANT
# ======================================================================

def add_subject_accuracy(dataframe):
    """
    Ajoute la précision globale de chaque participant.
    """
    audit_section(
        "CALCUL DE LA PRÉCISION PAR PARTICIPANT"
    )

    subject_summary = (
        dataframe
        .groupby(
            "subject_id",
            as_index=False,
        )
        .agg(
            subject_n_trials=(
                "sequence",
                "size",
            ),
            subject_n_accuracy_trials=(
                "is_correct",
                "count",
            ),
            subject_correct_count=(
                "is_correct",
                "sum",
            ),
            subject_accuracy=(
                "is_correct",
                "mean",
            ),
            subject_mean_confidence=(
                "confidence",
                "mean",
            ),
            subject_median_confidence=(
                "confidence",
                "median",
            ),
            subject_std_confidence=(
                "confidence",
                "std",
            ),
            subject_n_confidence_values=(
                "confidence",
                "nunique",
            ),
            subject_zero_confidence_rate=(
                "confidence",
                lambda values:
                    float(
                        (
                            values == 0
                        ).mean()
                    ),
            ),
            subject_hundred_confidence_rate=(
                "confidence",
                lambda values:
                    float(
                        (
                            values == 100
                        ).mean()
                    ),
            ),
            subject_condition_count=(
                "condition",
                "nunique",
            ),
            subject_condition=(
                "condition",
                "first",
            ),
        )
    )

    multiple_condition_subjects = (
        subject_summary.loc[
            subject_summary[
                "subject_condition_count"
            ] > 1
        ]
    )

    audit_print(
        "Participants :",
        len(subject_summary),
    )

    audit_print(
        "Participants apparaissant dans plusieurs conditions :",
        len(multiple_condition_subjects),
    )

    audit_print(
        "Précision moyenne entre participants :",
        round(
            subject_summary[
                "subject_accuracy"
            ].mean(),
            6,
        ),
    )

    audit_print(
        "Précision médiane entre participants :",
        round(
            subject_summary[
                "subject_accuracy"
            ].median(),
            6,
        ),
    )

    if (
        not multiple_condition_subjects.empty
        and STRICT_VALIDATION
    ):
        raise ValueError(
            "Certains participants apparaissent dans plusieurs "
            "conditions expérimentales."
        )

    dataframe = dataframe.merge(
        subject_summary[[
            "subject_id",
            "subject_n_trials",
            "subject_n_accuracy_trials",
            "subject_correct_count",
            "subject_accuracy",
            "subject_mean_confidence",
            "subject_median_confidence",
            "subject_std_confidence",
            "subject_n_confidence_values",
            "subject_zero_confidence_rate",
            "subject_hundred_confidence_rate",
            "subject_condition",
        ]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    return (
        dataframe,
        subject_summary,
    )


# ======================================================================
# FUSION DES COMPTAGES MREASONER
# ======================================================================

def merge_model_counts(
    experiment_data,
    model_data,
):
    """
    Fusionne les comptages mReasoner avec les essais humains.

    Clé :
        subject_id + task_type

    Le fichier mReasoner contient une ligne par participant et par
    structure formelle, tandis que le fichier expérimental contient
    plusieurs essais de chaque structure pour chaque participant.
    """
    audit_section(
        "FUSION DES COMPTAGES MREASONER"
    )

    model_columns = [
        "subject_id",
        "model_task_type",
        "model_task_index",
        "model_task_formal_normalized",
        "premise_1",
        "premise_2",
        "number_models_generated",
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",
        "n_parameter_sets_used",
        "epsilon",
        "lambda",
        "omega",
        "sigma",
    ]

    model_subset = model_data[
        model_columns
    ].copy()

    model_subset = model_subset.rename(
        columns={
            "model_task_type": "task_type",
            "premise_1": "model_premise_1",
            "premise_2": "model_premise_2",
        }
    )

    # Fusion gauche : toutes les observations humaines sont conservées.
    merged = experiment_data.merge(
        model_subset,
        on=[
            "subject_id",
            "task_type",
        ],
        how="left",
        validate="many_to_one",
        indicator="model_merge_status",
    )

    merge_counts = (
        merged[
            "model_merge_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    audit_print(
        "Résultat de la fusion :"
    )

    for status, count in merge_counts.items():
        audit_print(
            f"  {status} :",
            int(count),
        )

    matched_mask = (
        merged[
            "model_merge_status"
        ] == "both"
    )

    # Le dtype pandas "boolean" accepte trois états :
    # True, False et pd.NA.
    merged[
        "model_premises_match"
    ] = (
        merged[
            "task_formal_normalized"
        ]
        == merged[
            "model_task_formal_normalized"
        ]
    ).astype("boolean")

    # Les lignes non appariées n'ont pas de comparaison de prémisses
    # définie. Dans les données actuelles, il n'y en a aucune, puisque
    # les 9024 lignes sont appariées.
    merged.loc[
        ~matched_mask,
        "model_premises_match",
    ] = pd.NA


    premise_mismatch_mask = (
        matched_mask
        & (
            merged[
                "model_premises_match"
            ] != True
        )
    )

    premise_mismatch_count = int(
        premise_mismatch_mask.sum()
    )

    audit_print(
        "Essais appariés dont les prémisses ne correspondent pas :",
        premise_mismatch_count,
    )

    diagnostic_columns = [
        "subject_id",
        "sequence",
        "item_id",
        "task_type",
        "task_formal",
        "model_task_index",
        "model_premise_1",
        "model_premise_2",
        "number_models_generated",
        "model_merge_status",
        "model_premises_match",
    ]

    merged[
        diagnostic_columns
    ].to_csv(
        MODEL_MERGE_DIAGNOSTIC_FILE,
        index=False,
    )

    if (
        premise_mismatch_count > 0
        and STRICT_VALIDATION
    ):
        mismatch_preview = merged.loc[
            premise_mismatch_mask,
            diagnostic_columns,
        ].head(30)

        raise ValueError(
            "Certaines lignes ont été fusionnées sur le bon "
            "participant et le bon type, mais leurs prémisses "
            "ne correspondent pas.\n"
            f"{mismatch_preview.to_string(index=False)}"
        )

    missing_models = int(
        merged[
            "number_models_generated"
        ].isna().sum()
    )

    audit_print(
        "Essais sans nombre de modèles après fusion :",
        missing_models,
    )

    return merged


# ======================================================================
# AJOUT DES STATISTIQUES D'ITEM
# ======================================================================

def add_item_statistics(dataframe):
    """
    Calcule puis ajoute l'entropie et les statistiques collectives
    de chaque item.
    """
    audit_section(
        "CALCUL DE L'ENTROPIE PAR ITEM"
    )

    item_summary = compute_item_statistics(
        dataframe
    )

    item_summary.to_csv(
        ITEM_ENTROPY_SUMMARY_FILE,
        index=False,
    )

    audit_print(
        "Nombre d'items avec une entropie calculable :",
        int(
            item_summary[
                "item_entropy"
            ].notna().sum()
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

    dataframe = dataframe.merge(
        item_summary,
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    return (
        dataframe,
        item_summary,
    )


# ======================================================================
# DÉCOMPOSITION DU NOMBRE DE MODÈLES
# ======================================================================

def add_model_decomposition(dataframe):
    """
    Décompose le nombre de modèles en :

        1. moyenne du participant ;
        2. écart de chaque ligne à cette moyenne.

    Le nombre de modèles est actuellement constant pour une même
    combinaison participant × task_type. La variation intra-participant
    provient donc des différences entre MP, MT, AC et DA.
    """
    audit_section(
        "DÉCOMPOSITION DU NOMBRE DE MODÈLES"
    )

    subject_model_summary = (
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
            subject_median_models=(
                "number_models_generated",
                "median",
            ),
            subject_std_models_across_trials=(
                "number_models_generated",
                "std",
            ),
            subject_min_models=(
                "number_models_generated",
                "min",
            ),
            subject_max_models=(
                "number_models_generated",
                "max",
            ),
            subject_n_model_trials=(
                "number_models_generated",
                "count",
            ),
            subject_n_distinct_model_values=(
                "number_models_generated",
                "nunique",
            ),
        )
    )

    dataframe = dataframe.merge(
        subject_model_summary,
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
        "Moyenne des moyennes individuelles :",
        round(
            subject_model_summary[
                "subject_mean_models"
            ].mean(),
            6,
        ),
    )

    audit_print(
        "Écart-type des moyennes individuelles :",
        round(
            subject_model_summary[
                "subject_mean_models"
            ].std(),
            6,
        ),
    )

    zero_within_variation = int(
        (
            subject_model_summary[
                "subject_n_distinct_model_values"
            ] <= 1
        ).sum()
    )

    audit_print(
        "Participants sans variation du nombre de modèles "
        "entre leurs types de tâches :",
        zero_within_variation,
    )

    return (
        dataframe,
        subject_model_summary,
    )


# ======================================================================
# DIAGNOSTICS STATISTIQUES
# ======================================================================

def add_analysis_flags(dataframe):
    """
    Ajoute les indicateurs de complétude pour le futur modèle.
    """
    primary_columns = [
        "confidence",
        "subject_accuracy",
        "item_entropy",
        "number_models_generated",
        "subject_mean_models",
        "models_within_subject",
        "validity_binary",
        "condition",
        "subject_id",
        "item_id",
    ]

    dataframe[
        "analysis_complete"
    ] = (
        dataframe[
            primary_columns
        ]
        .notna()
        .all(axis=1)
    )

    missing_reasons = []

    for _, row in dataframe[
        primary_columns
    ].isna().iterrows():
        missing = [
            column
            for column in primary_columns
            if row[column]
        ]

        missing_reasons.append(
            ";".join(missing)
            if missing
            else ""
        )

    dataframe[
        "analysis_missing_reasons"
    ] = missing_reasons

    return dataframe


def compute_predictor_correlations(dataframe):
    """
    Calcule des corrélations descriptives entre variables numériques.

    Elles ne constituent pas encore le modèle statistique final.
    """
    numeric_columns = [
        "confidence",
        "is_correct",
        "subject_accuracy",
        "item_entropy",
        "item_accuracy",
        "item_yes_rate",
        "number_models_generated",
        "subject_mean_models",
        "models_within_subject",
        "std_models_generated",
        "validity_binary",
        "sequence",
        "rt",
        "logRT",
        "rt_for",
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in dataframe.columns
    ]

    numeric_data = dataframe[
        available_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation_matrix = numeric_data.corr(
        method="pearson"
    )

    correlation_matrix.to_csv(
        PREDICTOR_CORRELATIONS_FILE
    )

    return correlation_matrix


def print_missing_values(dataframe):
    """
    Affiche les valeurs manquantes des colonnes principales.
    """
    audit_section(
        "VALEURS MANQUANTES"
    )

    columns = [
        "subject_id",
        "sequence",
        "item_id",
        "task_type",
        "condition",
        "confidence",
        "is_correct",
        "subject_accuracy",
        "response_normalized",
        "validity_binary",
        "item_entropy",
        "item_accuracy",
        "number_models_generated",
        "std_models_generated",
        "subject_mean_models",
        "models_within_subject",
    ]

    available_columns = [
        column
        for column in columns
        if column in dataframe.columns
    ]

    missing_summary = (
        dataframe[
            available_columns
        ]
        .isna()
        .sum()
    )

    for column, count in missing_summary.items():
        percentage = (
            100.0
            * count
            / len(dataframe)
            if len(dataframe)
            else np.nan
        )

        audit_print(
            f"{column}:",
            f"{int(count)} manquant(s)",
            f"({percentage:.3f} %)",
        )


def print_confidence_diagnostics(dataframe):
    """
    Décrit la distribution de la confiance.
    """
    audit_section(
        "DISTRIBUTION DE LA CONFIANCE"
    )

    confidence = (
        pd.to_numeric(
            dataframe[
                "confidence"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if confidence.empty:
        audit_print(
            "Aucune confiance exploitable."
        )
        return

    audit_print(
        "Nombre de valeurs exploitables :",
        len(confidence),
    )

    audit_print(
        "Moyenne :",
        round(
            confidence.mean(),
            6,
        ),
    )

    audit_print(
        "Médiane :",
        round(
            confidence.median(),
            6,
        ),
    )

    audit_print(
        "Écart-type :",
        round(
            confidence.std(),
            6,
        ),
    )

    audit_print(
        "Minimum :",
        confidence.min(),
    )

    audit_print(
        "Maximum :",
        confidence.max(),
    )

    audit_print(
        "Nombre de valeurs distinctes :",
        confidence.nunique(),
    )

    zero_count = int(
        (
            confidence == 0
        ).sum()
    )

    hundred_count = int(
        (
            confidence == 100
        ).sum()
    )

    audit_print(
        "Valeurs égales à 0 :",
        zero_count,
        f"({100 * zero_count / len(confidence):.3f} %)",
    )

    audit_print(
        "Valeurs égales à 100 :",
        hundred_count,
        f"({100 * hundred_count / len(confidence):.3f} %)",
    )

    quantiles = confidence.quantile(
        [
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )

    audit_print(
        "Quantiles :"
    )

    for quantile, value in quantiles.items():
        audit_print(
            f"  q={quantile:.2f} :",
            round(
                float(value),
                4,
            ),
        )


def print_design_diagnostics(
    dataframe,
    item_summary,
):
    """
    Affiche les informations sur la structure participant × item.
    """
    audit_section(
        "STRUCTURE DU PLAN EXPÉRIMENTAL"
    )

    audit_print(
        "Nombre total de lignes :",
        len(dataframe),
    )

    audit_print(
        "Nombre de participants :",
        dataframe[
            "subject_id"
        ].nunique(),
    )

    audit_print(
        "Nombre d'items :",
        dataframe[
            "item_id"
        ].nunique(),
    )

    trials_per_subject = (
        dataframe
        .groupby(
            "subject_id"
        )
        .size()
    )

    audit_print(
        "Essais par participant — minimum :",
        int(
            trials_per_subject.min()
        ),
    )

    audit_print(
        "Essais par participant — médiane :",
        float(
            trials_per_subject.median()
        ),
    )

    audit_print(
        "Essais par participant — maximum :",
        int(
            trials_per_subject.max()
        ),
    )

    audit_print(
        "Participants par item — minimum :",
        int(
            item_summary[
                "item_n_subjects"
            ].min()
        ),
    )

    audit_print(
        "Participants par item — médiane :",
        float(
            item_summary[
                "item_n_subjects"
            ].median()
        ),
    )

    audit_print(
        "Participants par item — maximum :",
        int(
            item_summary[
                "item_n_subjects"
            ].max()
        ),
    )

    audit_print(
        "Répartition des conditions :"
    )

    condition_counts = (
        dataframe[[
            "subject_id",
            "condition",
        ]]
        .drop_duplicates()
        ["condition"]
        .value_counts(
            dropna=False
        )
    )

    for condition, count in condition_counts.items():
        audit_print(
            f"  {condition} :",
            int(count),
            "participant(s)",
        )

    audit_print(
        "Répartition des types de tâches :"
    )

    task_counts = (
        dataframe[
            "task_type"
        ]
        .value_counts(
            dropna=False
        )
    )

    for task_type, count in task_counts.items():
        audit_print(
            f"  {task_type} :",
            int(count),
            "essai(s)",
        )


def audit_validity_task_type_relation(
    dataframe,
):
    """
    Examine la relation entre validité et type de tâche.
    """
    audit_section(
        "RELATION ENTRE VALIDITÉ ET TYPE DE TÂCHE"
    )

    contingency = pd.crosstab(
        dataframe[
            "task_type"
        ],
        dataframe[
            "validity_binary"
        ],
        dropna=False,
    )

    audit_print(
        contingency.to_string()
    )

    validity_count_by_task = (
        dataframe
        .groupby(
            "task_type"
        )[
            "validity_binary"
        ]
        .nunique(
            dropna=True
        )
    )

    deterministic = bool(
        (
            validity_count_by_task <= 1
        ).all()
    )

    audit_print(
        "La validité est-elle constante à l'intérieur "
        "de chaque task_type ?",
        deterministic,
    )

    if deterministic:
        audit_print(
            "ATTENTION : validité et task_type sont structurellement "
            "liés. Ils ne devront pas être introduits simultanément "
            "sans précaution dans le même modèle."
        )


def audit_model_simulation_stability(
    dataframe,
):
    """
    Décrit la stabilité du nombre de modèles sur les simulations.
    """
    audit_section(
        "STABILITÉ DES SIMULATIONS MREASONER"
    )

    model_rows = (
        dataframe[[
            "subject_id",
            "task_type",
            "number_models_generated",
            "std_models_generated",
            "minimum_models_generated",
            "maximum_models_generated",
            "n_samples",
        ]]
        .drop_duplicates(
            subset=[
                "subject_id",
                "task_type",
                ]
        )
        .copy()
    )

    model_rows[
        "model_range"
    ] = (
        model_rows[
            "maximum_models_generated"
        ]
        - model_rows[
            "minimum_models_generated"
        ]
    )

    usable_std = model_rows[
        "std_models_generated"
    ].dropna()

    if usable_std.empty:
        audit_print(
            "Aucune information de stabilité disponible."
        )
        return

    audit_print(
        "Nombre de combinaisons participant × tâche :",
        len(model_rows),
    )

    audit_print(
        "Nombre de simulations observé :",
        sorted(
            model_rows[
                "n_samples"
            ]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    audit_print(
        "Écart-type moyen des simulations :",
        round(
            usable_std.mean(),
            6,
        ),
    )

    audit_print(
        "Écart-type médian des simulations :",
        round(
            usable_std.median(),
            6,
        ),
    )

    stable_count = int(
        np.isclose(
            usable_std,
            0.0,
        ).sum()
    )

    audit_print(
        "Combinaisons avec écart-type nul :",
        stable_count,
        (
            f"({100 * stable_count / len(usable_std):.3f} %)"
        ),
    )

    large_range_count = int(
        (
            model_rows[
                "model_range"
            ] >= 2
        ).sum()
    )

    audit_print(
        "Combinaisons avec amplitude max-min >= 2 :",
        large_range_count,
        (
            f"({100 * large_range_count / len(model_rows):.3f} %)"
        ),
    )


# ======================================================================
# ORDRE DES COLONNES
# ======================================================================

def reorder_columns(dataframe):
    """
    Place les colonnes analytiques principales au début du fichier.
    """
    preferred_columns = [
        # Identifiants
        "subject_id",
        "sequence",
        "item_id",
        "qnum",

        # Variable dépendante
        "confidence",

        # Quatre concepts principaux
        "subject_accuracy",
        "item_entropy",
        "number_models_generated",
        "validity_binary",

        # Décomposition du nombre de modèles
        "subject_mean_models",
        "models_within_subject",

        # Variables humaines
        "response",
        "response_normalized",
        "response_binary",
        "is_correct",

        # Statistiques d'item
        "item_n_rows",
        "item_n_responses",
        "item_n_subjects",
        "item_yes_count",
        "item_no_count",
        "item_yes_rate",
        "item_no_rate",
        "item_accuracy",

        # Description expérimentale
        "task_type",
        "task_formal",
        "condition",
        "validity",
        "believability",
        "conflict",
        "stimulus",

        # Modèles mentaux
        "std_models_generated",
        "minimum_models_generated",
        "maximum_models_generated",
        "n_samples",
        "n_parameter_sets_used",

        # Paramètres mReasoner
        "epsilon",
        "lambda",
        "omega",
        "sigma",

        # Résumés participant
        "subject_n_trials",
        "subject_n_accuracy_trials",
        "subject_correct_count",
        "subject_mean_confidence",
        "subject_median_confidence",
        "subject_std_confidence",
        "subject_n_confidence_values",
        "subject_zero_confidence_rate",
        "subject_hundred_confidence_rate",
        "subject_condition",
        "subject_median_models",
        "subject_std_models_across_trials",
        "subject_min_models",
        "subject_max_models",
        "subject_n_model_trials",
        "subject_n_distinct_model_values",

        # Temps
        "rt",
        "logRT",
        "rt_for",

        # Contrôles de fusion
        "model_task_index",
        "model_merge_status",
        "model_premises_match",

        # Complétude
        "analysis_complete",
        "analysis_missing_reasons",
    ]

    existing_preferred = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    remaining_columns = [
        column
        for column in dataframe.columns
        if column not in existing_preferred
    ]

    return dataframe[
        existing_preferred
        + remaining_columns
    ]


# ======================================================================
# PROGRAMME PRINCIPAL
# ======================================================================

def main():
    print("=" * 80)
    print("CONSTRUCTION DU DATASET ANALYTIQUE E1")
    print("=" * 80)

    try:
        # ==============================================================
        # 1. Chargement
        # ==============================================================

        experiment_data = (
            load_experiment_data()
        )

        model_data = (
            load_model_count_data()
        )

        # ==============================================================
        # 2. Validation de la structure des items
        # ==============================================================

        audit_item_consistency(
            experiment_data
        )

        # ==============================================================
        # 3. Précision par participant
        # ==============================================================

        (
            experiment_data,
            subject_summary,
        ) = add_subject_accuracy(
            experiment_data
        )

        # ==============================================================
        # 4. Entropie et statistiques par item
        # ==============================================================

        (
            experiment_data,
            item_summary,
        ) = add_item_statistics(
            experiment_data
        )

        # ==============================================================
        # 5. Fusion des comptages mReasoner
        # ==============================================================

        analysis_data = merge_model_counts(
            experiment_data=experiment_data,
            model_data=model_data,
        )

        # ==============================================================
        # 6. Décomposition intra/interindividuelle des modèles
        # ==============================================================

        (
            analysis_data,
            subject_model_summary,
        ) = add_model_decomposition(
            analysis_data
        )

        # Ajoute les données de modèles au résumé participant.
        subject_summary = subject_summary.merge(
            subject_model_summary,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )

        subject_summary.to_csv(
            SUBJECT_SUMMARY_FILE,
            index=False,
        )

        # ==============================================================
        # 7. Indicateur de complétude
        # ==============================================================

        analysis_data = add_analysis_flags(
            analysis_data
        )

        # ==============================================================
        # 8. Diagnostics
        # ==============================================================

        print_missing_values(
            analysis_data
        )

        print_confidence_diagnostics(
            analysis_data
        )

        print_design_diagnostics(
            dataframe=analysis_data,
            item_summary=item_summary,
        )

        audit_validity_task_type_relation(
            analysis_data
        )

        audit_model_simulation_stability(
            analysis_data
        )

        correlation_matrix = (
            compute_predictor_correlations(
                analysis_data
            )
        )

        audit_section(
            "CORRÉLATIONS DESCRIPTIVES"
        )

        audit_print(
            correlation_matrix
            .round(4)
            .to_string()
        )

        # ==============================================================
        # 9. Tri et sauvegarde
        # ==============================================================

        analysis_data = (
            analysis_data
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

        analysis_data = reorder_columns(
            analysis_data
        )

        if not KEEP_INCOMPLETE_ROWS:
            analysis_data = analysis_data.loc[
                analysis_data[
                    "analysis_complete"
                ]
            ].copy()

        analysis_data.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        # ==============================================================
        # 10. Résumé final
        # ==============================================================

        audit_section(
            "RÉSUMÉ FINAL"
        )

        audit_print(
            "Fichier analytique créé :",
            OUTPUT_FILE,
        )

        audit_print(
            "Nombre de lignes finales :",
            len(analysis_data),
        )

        audit_print(
            "Nombre de participants :",
            analysis_data[
                "subject_id"
            ].nunique(),
        )

        audit_print(
            "Nombre d'items :",
            analysis_data[
                "item_id"
            ].nunique(),
        )

        complete_count = int(
            analysis_data[
                "analysis_complete"
            ].sum()
        )

        audit_print(
            "Lignes complètes pour le modèle principal :",
            complete_count,
        )

        audit_print(
            "Lignes incomplètes pour le modèle principal :",
            int(
                len(analysis_data)
                - complete_count
            ),
        )

        if len(analysis_data):
            audit_print(
                "Pourcentage de lignes complètes :",
                f"{100 * complete_count / len(analysis_data):.3f} %",
            )

        audit_print(
            "Résumé des items :",
            ITEM_ENTROPY_SUMMARY_FILE,
        )

        audit_print(
            "Résumé des participants :",
            SUBJECT_SUMMARY_FILE,
        )

        audit_print(
            "Diagnostic de fusion :",
            MODEL_MERGE_DIAGNOSTIC_FILE,
        )

        audit_print(
            "Corrélations descriptives :",
            PREDICTOR_CORRELATIONS_FILE,
        )

        audit_print(
            "Contrôle des items :",
            ITEM_CONSISTENCY_FILE,
        )

        audit_print(
            "Structure du fichier de modèles :",
            MODEL_STRUCTURE_FILE,
        )

        audit_print(
            "\nAperçu du fichier analytique :"
        )

        audit_print(
            analysis_data
            .head(10)
            .to_string(
                index=False
            )
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
