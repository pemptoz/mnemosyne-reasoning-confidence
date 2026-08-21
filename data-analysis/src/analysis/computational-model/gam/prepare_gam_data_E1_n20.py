#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_gam_data_E1_n20.py

Prépare les données nécessaires au GAM de confiance pour
l'expérience E1.

Ce script :

    1. charge dataset_analysis_E1_n20.csv ;
    2. vérifie les colonnes requises ;
    3. conserve les lignes complètes ;
    4. contrôle les valeurs de confidence, condition et validity_binary ;
    5. centre la séquence ;
    6. standardise les prédicteurs continus ;
    7. encode condition, subject_id et item_id ;
    8. construit la matrice X et la variable dépendante y.

Aucun modèle n'est ajusté dans ce script.
Aucun fichier de résultat n'est créé.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# correspond à la racine du projet.
PROJECT_ROOT = SCRIPT_DIR.parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "gam"
    / "dataset_analysis_E1_n20.csv"
)


# ============================================================================
# COLONNES NÉCESSAIRES
# ============================================================================

REQUIRED_COLUMNS = [
    "confidence",
    "condition",
    "sequence",
    "subject_id",
    "item_id",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
    "validity_binary",
]

NUMERIC_COLUMNS = [
    "confidence",
    "sequence",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
    "validity_binary",
]

STANDARDIZED_COLUMNS = [
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]


# ============================================================================
# ORDRE DES VARIABLES DANS LA MATRICE X
# ============================================================================

FEATURE_COLUMNS = [
    "condition_code",
    "sequence_c10",
    "subject_accuracy_z",
    "item_entropy_z",
    "subject_mean_models_z",
    "models_within_subject_z",
    "validity_binary",
    "subject_code",
    "item_code",
]

FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(FEATURE_COLUMNS)
}


# ============================================================================
# OUTILS
# ============================================================================

def section(title):
    """Affiche un titre de section."""
    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_identifier(value):
    """
    Normalise un identifiant de participant ou d'item.

    Exemples :
        63873.0 -> "63873"
        63873   -> "63873"
    """
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


def normalize_analysis_complete(value):
    """
    Convertit les différentes représentations possibles d'une valeur
    vraie en booléen.
    """
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    normalized = str(value).strip().lower()

    return normalized in {
        "true",
        "1",
        "1.0",
        "yes",
        "oui",
    }


def standardize(series):
    """
    Standardise une variable numérique avec :

        z = (x - moyenne) / écart-type

    L'écart-type empirique est calculé avec ddof=1.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean = float(numeric.mean())

    standard_deviation = float(
        numeric.std(ddof=1)
    )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        raise ValueError(
            f"Impossible de standardiser {series.name} : "
            f"écart-type = {standard_deviation}"
        )

    standardized = (
        numeric - mean
    ) / standard_deviation

    return (
        standardized,
        mean,
        standard_deviation,
    )


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def load_data():
    """
    Charge le dataset analytique et vérifie les colonnes nécessaires.
    """
    section(
        "CHARGEMENT DES DONNÉES"
    )

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE
    )

    print("Fichier :", DATA_FILE)
    print("Nombre de lignes brutes :", len(data))
    print("Nombre de colonnes :", len(data.columns))

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            "Colonnes nécessaires absentes : "
            + ", ".join(missing_columns)
        )

    return data.copy()


# ============================================================================
# PRÉPARATION DES DONNÉES
# ============================================================================

def prepare_data(data):
    """
    Nettoie les données et construit les variables utilisées par le GAM.
    """
    section(
        "PRÉPARATION DES DONNÉES"
    )

    # ------------------------------------------------------------------
    # 1. Filtrage par analysis_complete
    # ------------------------------------------------------------------

    if "analysis_complete" in data.columns:
        complete_mask = (
            data["analysis_complete"]
            .apply(normalize_analysis_complete)
        )

        before_filter = len(data)

        data = data.loc[
            complete_mask
        ].copy()

        print(
            "Lignes retirées car analysis_complete=False :",
            before_filter - len(data),
        )

    # ------------------------------------------------------------------
    # 2. Conversion des variables numériques
    # ------------------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------
    # 3. Normalisation des identifiants
    # ------------------------------------------------------------------

    data["subject_id"] = (
        data["subject_id"]
        .apply(normalize_identifier)
        .astype("string")
    )

    data["item_id"] = (
        data["item_id"]
        .apply(normalize_identifier)
        .astype("string")
    )

    # ------------------------------------------------------------------
    # 4. Normalisation de la condition
    # ------------------------------------------------------------------

    data["condition"] = (
        data["condition"]
        .astype("string")
        .str.strip()
    )

    # ------------------------------------------------------------------
    # 5. Suppression des valeurs non finies et manquantes
    # ------------------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before_drop = len(data)

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    print(
        "Lignes supprimées pour donnée essentielle manquante :",
        before_drop - len(data),
    )

    # ------------------------------------------------------------------
    # 6. Vérification de confidence
    # ------------------------------------------------------------------

    invalid_confidence = (
        (data["confidence"] < 0)
        | (data["confidence"] > 100)
    )

    if invalid_confidence.any():
        raise ValueError(
            "Certaines valeurs de confidence sont hors de [0, 100]."
        )

    # ------------------------------------------------------------------
    # 7. Vérification des conditions
    # ------------------------------------------------------------------

    expected_conditions = {
        "Neutral",
        "Standard",
    }

    observed_conditions = set(
        data["condition"].unique()
    )

    if observed_conditions != expected_conditions:
        raise ValueError(
            "Les conditions attendues sont Neutral et Standard. "
            f"Conditions observées : {sorted(observed_conditions)}"
        )

    # ------------------------------------------------------------------
    # 8. Vérification de validity_binary
    # ------------------------------------------------------------------

    validity_values = set(
        data["validity_binary"]
        .dropna()
        .unique()
    )

    if not validity_values.issubset({
        0,
        1,
        0.0,
        1.0,
    }):
        raise ValueError(
            "validity_binary doit uniquement contenir 0 et 1. "
            f"Valeurs observées : {sorted(validity_values)}"
        )

    if validity_values != {0.0, 1.0}:
        raise ValueError(
            "Les deux niveaux de validity_binary doivent être présents."
        )

    data["validity_binary"] = (
        data["validity_binary"]
        .astype(int)
    )

    # ------------------------------------------------------------------
    # 9. Centrage et mise à l'échelle de sequence
    # ------------------------------------------------------------------

    sequence_mean = float(
        data["sequence"].mean()
    )

    data["sequence_c10"] = (
        data["sequence"]
        - sequence_mean
    ) / 10.0

    print(
        "Moyenne utilisée pour centrer sequence :",
        sequence_mean,
    )

    # ------------------------------------------------------------------
    # 10. Standardisation des prédicteurs continus
    # ------------------------------------------------------------------

    standardization_rows = []

    for column in STANDARDIZED_COLUMNS:
        standardized_column = (
            f"{column}_z"
        )

        (
            data[standardized_column],
            variable_mean,
            variable_standard_deviation,
        ) = standardize(
            data[column]
        )

        standardization_rows.append({
            "variable":
                column,

            "standardized_variable":
                standardized_column,

            "mean":
                variable_mean,

            "standard_deviation":
                variable_standard_deviation,
        })

    standardization_table = pd.DataFrame(
        standardization_rows
    )

    # ------------------------------------------------------------------
    # 11. Encodage de la condition
    # ------------------------------------------------------------------

    condition_mapping = {
        "Neutral": 0,
        "Standard": 1,
    }

    data["condition_code"] = (
        data["condition"]
        .map(condition_mapping)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # 12. Encodage des participants
    # ------------------------------------------------------------------

    subject_categories = sorted(
        data["subject_id"]
        .astype(str)
        .unique()
    )

    subject_mapping = {
        subject_id: code
        for code, subject_id
        in enumerate(subject_categories)
    }

    data["subject_code"] = (
        data["subject_id"]
        .astype(str)
        .map(subject_mapping)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # 13. Encodage des items
    # ------------------------------------------------------------------

    item_categories = sorted(
        data["item_id"]
        .astype(str)
        .unique()
    )

    item_mapping = {
        item_id: code
        for code, item_id
        in enumerate(item_categories)
    }

    data["item_code"] = (
        data["item_id"]
        .astype(str)
        .map(item_mapping)
        .astype(int)
    )

    # ------------------------------------------------------------------
    # 14. Tri stable
    # ------------------------------------------------------------------

    data = (
        data
        .sort_values(
            by=[
                "subject_id",
                "sequence",
            ]
        )
        .reset_index(drop=True)
    )

    preparation_information = {
        "sequence_mean":
            sequence_mean,

        "condition_mapping":
            condition_mapping,

        "subject_mapping":
            subject_mapping,

        "item_mapping":
            item_mapping,

        "standardization_table":
            standardization_table,
    }

    return (
        data,
        preparation_information,
    )


# ============================================================================
# CONSTRUCTION DE X ET y
# ============================================================================

def build_model_matrices(data):
    """
    Construit :

        X : matrice des prédicteurs ;
        y : variable dépendante confidence.

    L'ordre des colonnes de X est défini par FEATURE_COLUMNS.
    """
    section(
        "CONSTRUCTION DE X ET y"
    )

    X = data[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    y = data[
        "confidence"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "La matrice X contient des valeurs non finies."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "La variable y contient des valeurs non finies."
        )

    if len(X) != len(y):
        raise ValueError(
            "X et y ne contiennent pas le même nombre d'observations."
        )

    print("Dimensions de X :", X.shape)
    print("Dimensions de y :", y.shape)

    print("")
    print("Ordre des variables dans X :")

    for feature_name, feature_index in FEATURE_INDEX.items():
        print(
            f"  Colonne {feature_index} : {feature_name}"
        )

    return X, y


# ============================================================================
# CONTRÔLES FINAUX
# ============================================================================

def print_final_checks(
    data,
    X,
    y,
    preparation_information,
):
    """
    Affiche les principaux contrôles de préparation.
    """
    section(
        "CONTRÔLES FINAUX"
    )

    print(
        "Nombre d'observations :",
        len(data),
    )

    print(
        "Nombre de participants :",
        data["subject_id"].nunique(),
    )

    print(
        "Nombre d'items :",
        data["item_id"].nunique(),
    )

    print(
        "Conditions :",
        sorted(data["condition"].unique()),
    )

    print(
        "Valeurs de validity_binary :",
        sorted(data["validity_binary"].unique()),
    )

    print(
        "Confiance moyenne :",
        round(float(y.mean()), 4),
    )

    print(
        "Confiance minimale :",
        float(y.min()),
    )

    print(
        "Confiance maximale :",
        float(y.max()),
    )

    print("")
    print("Paramètres de standardisation :")

    print(
        preparation_information[
            "standardization_table"
        ].to_string(index=False)
    )

    print("")
    print("Aperçu des données préparées :")

    preview_columns = [
        "subject_id",
        "item_id",
        "condition",
        "condition_code",
        "sequence",
        "sequence_c10",
        "subject_accuracy_z",
        "item_entropy_z",
        "subject_mean_models_z",
        "models_within_subject_z",
        "validity_binary",
        "subject_code",
        "item_code",
        "confidence",
    ]

    print(
        data[
            preview_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print("")
    print(
        "Préparation terminée : aucun GAM n'a encore été ajusté."
    )


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main():
    section(
        "PRÉPARATION DES DONNÉES DU GAM — EXPÉRIENCE E1"
    )

    data = load_data()

    (
        data,
        preparation_information,
    ) = prepare_data(
        data
    )

    X, y = build_model_matrices(
        data
    )

    print_final_checks(
        data=data,
        X=X,
        y=y,
        preparation_information=
            preparation_information,
    )


if __name__ == "__main__":
    main()
