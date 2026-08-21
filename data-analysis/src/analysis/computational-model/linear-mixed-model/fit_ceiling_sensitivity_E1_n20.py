#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_ceiling_sensitivity_E1.py

Analyse de sensibilité du modèle cognitif final après exclusion des
réponses de confiance égales à 100.

Fichier d'entrée
----------------
results/tables/computational-model/dataset_analysis_E1_n20.csv

Objectif
--------
Vérifier si les associations du modèle final restent observables
lorsque les réponses situées à la borne supérieure de l'échelle sont
retirées.

Déroulement
-----------
1. Chargement du dataset complet.
2. Centrage de sequence sur le dataset complet.
3. Standardisation des prédicteurs sur le dataset complet.
4. Exclusion des observations avec confidence = 100.
5. Ajustement du modèle linéaire mixte en REML.

Modèle
------
confidence ~
    condition
    + sequence_c10
    + subject_accuracy_z
    + item_entropy_z
    + subject_mean_models_z
    + models_within_subject_z

La validité n'est plus incluse, car l'analyse de sensibilité
précédente a montré qu'elle n'améliorait pas clairement le modèle.

Effets aléatoires croisés
-------------------------
- intercept participant ;
- intercept item.

Fichiers produits
-----------------
ceiling_sensitivity_E1_n20/
    ceiling_summary.csv
    below_ceiling_fixed_effects.csv
    below_ceiling_REML_summary.txt
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "computational-model"
    / "dataset_analysis_E1_n20.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "computational-model"
    / "linear-mixed-model"
    / "ceiling_sensitivity_E1_n20"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------
# Fichiers produits
# --------------------------------------------------------------------------

CEILING_SUMMARY_FILE = (
    OUTPUT_DIR
    / "ceiling_summary.csv"
)

FIXED_EFFECTS_FILE = (
    OUTPUT_DIR
    / "below_ceiling_fixed_effects.csv"
)

REML_SUMMARY_FILE = (
    OUTPUT_DIR
    / "below_ceiling_REML_summary.txt"
)


# --------------------------------------------------------------------------
# Formule du modèle
# --------------------------------------------------------------------------

FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z"
)


# --------------------------------------------------------------------------
# Effets aléatoires croisés
# --------------------------------------------------------------------------

VC_FORMULA = {
    "item":
        "0 + C(item_id)",

    "subject":
        "0 + C(subject_id)",
}


# --------------------------------------------------------------------------
# Colonnes nécessaires
# --------------------------------------------------------------------------

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
]

STANDARDIZED_COLUMNS = [
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]


# --------------------------------------------------------------------------
# Options d'optimisation
# --------------------------------------------------------------------------

OPTIMIZERS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]

MAX_ITERATIONS = 3000

USE_SPARSE_MATRICES = True


# ============================================================================
# AFFICHAGE
# ============================================================================

def section(title):
    """Affiche un titre de section."""
    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# OUTILS GÉNÉRAUX
# ============================================================================

def safe_float(value):
    """
    Convertit une valeur en float.

    Retourne NaN si la conversion est impossible ou si la valeur
    n'est pas finie.
    """
    try:
        numeric = float(value)

    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(numeric):
        return np.nan

    return numeric


def normalize_identifier(value):
    """
    Normalise un identifiant participant ou item.

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


def normalize_condition(value):
    """
    Normalise la condition vers Neutral ou Standard.
    """
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().lower()

    if normalized in {
        "neutral",
        "neutre",
    }:
        return "Neutral"

    if normalized == "standard":
        return "Standard"

    return pd.NA


def standardize(series):
    """
    Standardise une variable :

        z = (x - moyenne) / écart-type

    L'écart-type empirique est calculé avec ddof=1.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean = float(
        numeric.mean()
    )

    standard_deviation = float(
        numeric.std(ddof=1)
    )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        raise ValueError(
            "Impossible de standardiser "
            f"{series.name} : "
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
# CHARGEMENT ET PRÉPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Charge le dataset complet et construit les transformations du
    modèle final.

    Le centrage et la standardisation sont effectués avant le retrait
    des réponses égales à 100.
    """
    section(
        "CHARGEMENT ET PRÉPARATION DES DONNÉES"
    )

    print(
        "Racine du projet :",
        PROJECT_ROOT,
    )

    print(
        "Fichier de données :",
        DATA_FILE,
    )

    print(
        "Dossier de sortie :",
        OUTPUT_DIR,
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE
    )

    print(
        "Nombre de lignes brutes :",
        len(data),
    )

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

    data = data.copy()

    # ------------------------------------------------------------------
    # Filtrage sur analysis_complete
    # ------------------------------------------------------------------

    if "analysis_complete" in data.columns:
        complete_mask = (
            data["analysis_complete"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({
                "true",
                "1",
                "1.0",
                "yes",
            })
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
    # Conversion numérique
    # ------------------------------------------------------------------

    numeric_columns = [
        "confidence",
        "sequence",
        "subject_accuracy",
        "item_entropy",
        "subject_mean_models",
        "models_within_subject",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------
    # Normalisation des identifiants et de la condition
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

    data["condition"] = (
        data["condition"]
        .apply(normalize_condition)
        .astype("string")
    )

    # ------------------------------------------------------------------
    # Suppression des données essentielles manquantes
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
    # Contrôle de la confiance
    # ------------------------------------------------------------------

    invalid_confidence = (
        (data["confidence"] < 0)
        | (data["confidence"] > 100)
    )

    if invalid_confidence.any():
        raise ValueError(
            "Certaines valeurs de confiance sont hors de [0, 100]."
        )

    # ------------------------------------------------------------------
    # Contrôle des conditions
    # ------------------------------------------------------------------

    expected_conditions = {
        "Neutral",
        "Standard",
    }

    observed_conditions = set(
        data["condition"]
        .dropna()
        .unique()
    )

    if observed_conditions != expected_conditions:
        raise ValueError(
            "Conditions attendues : Neutral et Standard. "
            "Conditions observées : "
            f"{sorted(observed_conditions)}"
        )

    # ------------------------------------------------------------------
    # Vérification du plan entre participants
    # ------------------------------------------------------------------

    condition_counts_by_subject = (
        data
        .groupby("subject_id")
        ["condition"]
        .nunique()
    )

    subjects_in_multiple_conditions = (
        condition_counts_by_subject[
            condition_counts_by_subject > 1
        ]
    )

    if not subjects_in_multiple_conditions.empty:
        raise ValueError(
            "Certains participants apparaissent dans plusieurs "
            "conditions."
        )

    if data["subject_id"].nunique() < 2:
        raise ValueError(
            "Au moins deux participants sont nécessaires."
        )

    if data["item_id"].nunique() < 2:
        raise ValueError(
            "Au moins deux items sont nécessaires."
        )

    # ------------------------------------------------------------------
    # Centrage de la séquence sur le dataset complet
    # ------------------------------------------------------------------

    sequence_mean = float(
        data["sequence"].mean()
    )

    data["sequence_c10"] = (
        data["sequence"]
        - sequence_mean
    ) / 10.0

    # ------------------------------------------------------------------
    # Standardisation sur le dataset complet
    # ------------------------------------------------------------------

    standardization_rows = []

    for variable in STANDARDIZED_COLUMNS:
        standardized_name = (
            f"{variable}_z"
        )

        (
            data[standardized_name],
            variable_mean,
            variable_standard_deviation,
        ) = standardize(
            data[variable]
        )

        standardization_rows.append({
            "variable":
                variable,

            "standardized_variable":
                standardized_name,

            "mean":
                variable_mean,

            "standard_deviation":
                variable_standard_deviation,
        })

    standardization_table = pd.DataFrame(
        standardization_rows
    )

    # ------------------------------------------------------------------
    # Indicateur de plafond
    # ------------------------------------------------------------------

    if "is_ceiling" in data.columns:
        provided_is_ceiling = pd.to_numeric(
            data["is_ceiling"],
            errors="coerce",
        )

        calculated_is_ceiling = (
            data["confidence"] == 100
        ).astype(int)

        inconsistent_ceiling = (
            provided_is_ceiling.notna()
            & (
                provided_is_ceiling
                != calculated_is_ceiling
            )
        )

        if inconsistent_ceiling.any():
            raise ValueError(
                "La colonne is_ceiling n'est pas cohérente "
                "avec confidence == 100."
            )

    data["at_ceiling"] = (
        data["confidence"] == 100
    ).astype(int)

    # ------------------------------------------------------------------
    # Tri stable
    # ------------------------------------------------------------------

    data = (
        data
        .sort_values(
            by=[
                "subject_id",
                "item_id",
                "sequence",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Nombre de lignes exploitables avant exclusion :",
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
        "Centre de la séquence :",
        sequence_mean,
    )

    print("")
    print(
        "Paramètres de standardisation :"
    )

    print(
        standardization_table
        .to_string(index=False)
    )

    return (
        data,
        sequence_mean,
        standardization_table,
    )


# ============================================================================
# RÉSUMÉ DU PLAFOND
# ============================================================================

def create_ceiling_summary(
    full_data,
    below_ceiling_data,
):
    """
    Résume la fréquence du plafond et l'échantillon restant.
    """
    return pd.DataFrame([
        {
            "n_total":
                int(
                    len(full_data)
                ),

            "n_at_ceiling":
                int(
                    full_data[
                        "at_ceiling"
                    ].sum()
                ),

            "ceiling_rate":
                float(
                    full_data[
                        "at_ceiling"
                    ].mean()
                ),

            "n_below_ceiling":
                int(
                    len(
                        below_ceiling_data
                    )
                ),

            "n_subjects_below_ceiling":
                int(
                    below_ceiling_data[
                        "subject_id"
                    ].nunique()
                ),

            "n_items_below_ceiling":
                int(
                    below_ceiling_data[
                        "item_id"
                    ].nunique()
                ),
        }
    ])


# ============================================================================
# CONSTRUCTION DU MODÈLE
# ============================================================================

def build_mixed_model(data):
    """
    Construit le modèle linéaire mixte avec des intercepts aléatoires
    croisés pour les participants et les items.
    """
    model_data = data.copy()

    model_data["_global_group"] = (
        "all_observations"
    )

    return smf.mixedlm(
        formula=FORMULA,
        data=model_data,
        groups=model_data[
            "_global_group"
        ],
        re_formula="0",
        vc_formula=VC_FORMULA,
        use_sparse=USE_SPARSE_MATRICES,
    )


# ============================================================================
# AJUSTEMENT REML
# ============================================================================

def fit_reml_model(data):
    """
    Ajuste le modèle en REML en essayant plusieurs optimiseurs.

    Le premier ajustement signalé comme convergé est conservé.
    """
    section(
        "AJUSTEMENT DU MODÈLE SOUS LE PLAFOND — REML"
    )

    print(
        "Formule :",
        FORMULA,
    )

    last_error = None
    last_result = None

    for optimizer in OPTIMIZERS:
        print("")
        print(
            "Optimiseur essayé :",
            optimizer,
        )

        try:
            model = build_mixed_model(
                data
            )

            with warnings.catch_warnings(
                record=True
            ) as caught_warnings:
                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    reml=True,
                    method=optimizer,
                    maxiter=MAX_ITERATIONS,
                    full_output=True,
                    disp=False,
                )

            last_result = result

            for warning in caught_warnings:
                print(
                    "Avertissement statsmodels :",
                    warning.message,
                )

            converged = bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            )

            print(
                "Convergence :",
                converged,
            )

            print(
                "Log-vraisemblance :",
                safe_float(result.llf),
            )

            if converged:
                print(
                    "Optimiseur retenu :",
                    optimizer,
                )

                return (
                    result,
                    optimizer,
                )

        except Exception as error:
            last_error = error

            print(
                "Échec avec",
                optimizer,
                ":",
                repr(error),
            )

    if last_result is not None:
        raise RuntimeError(
            "Un résultat a été obtenu, mais aucun optimiseur "
            "n'a signalé une convergence complète."
        )

    raise RuntimeError(
        "Impossible d'ajuster le modèle sous le plafond. "
        f"Dernière erreur : {last_error!r}"
    )


# ============================================================================
# EFFETS FIXES
# ============================================================================

def create_fixed_effects_table(result):
    """
    Extrait les effets fixes et calcule les tests de Wald bilatéraux.

    Pour chaque coefficient :

        z = estimation / erreur standard

        p = 2 × P(Z > |z|)

    sous l'approximation normale lorsque l'hypothèse beta = 0
    est vraie.
    """
    parameter_names = list(
        result.fe_params.index
    )

    estimates = np.asarray(
        result.fe_params,
        dtype=float,
    )

    covariance_matrix = (
        result
        .cov_params()
        .loc[
            parameter_names,
            parameter_names,
        ]
    )

    standard_errors = np.sqrt(
        np.diag(
            covariance_matrix
        )
    )

    z_values = (
        estimates
        / standard_errors
    )

    p_values = (
        2
        * stats.norm.sf(
            np.abs(z_values)
        )
    )

    return pd.DataFrame({
        "parameter":
            parameter_names,

        "estimate":
            estimates,

        "standard_error":
            standard_errors,

        "z_value":
            z_values,

        "p_value":
            p_values,

        "ci_95_lower":
            estimates
            - 1.96
            * standard_errors,

        "ci_95_upper":
            estimates
            + 1.96
            * standard_errors,
    })


# ============================================================================
# SAUVEGARDE DU RÉSUMÉ
# ============================================================================

def save_reml_summary(
    result,
    optimizer,
    full_data,
    below_ceiling_data,
    sequence_mean,
    standardization_table,
):
    """
    Sauvegarde le résumé complet de l'analyse sous le plafond.
    """
    with open(
        REML_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "ANALYSE DE SENSIBILITÉ SOUS LE PLAFOND E1 — REML\n"
        )

        output_file.write(
            "=" * 80
        )

        output_file.write("\n\n")

        output_file.write(
            "ÉCHANTILLON\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            f"Observations initiales : {len(full_data)}\n"
        )

        output_file.write(
            "Observations avec confidence = 100 : "
            f"{int(full_data['at_ceiling'].sum())}\n"
        )

        output_file.write(
            "Proportion avec confidence = 100 : "
            f"{full_data['at_ceiling'].mean():.6f}\n"
        )

        output_file.write(
            "Observations inférieures à 100 : "
            f"{len(below_ceiling_data)}\n"
        )

        output_file.write(
            "Participants représentés sous le plafond : "
            f"{below_ceiling_data['subject_id'].nunique()}\n"
        )

        output_file.write(
            "Items représentés sous le plafond : "
            f"{below_ceiling_data['item_id'].nunique()}\n"
        )

        output_file.write("\n")

        output_file.write(
            "FORMULE\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            FORMULA
        )

        output_file.write("\n\n")

        output_file.write(
            f"Optimiseur retenu : {optimizer}\n"
        )

        output_file.write(
            f"Centre de la séquence : {sequence_mean:.6f}\n"
        )

        output_file.write("\n")

        output_file.write(
            "STANDARDISATION SUR LE DATASET COMPLET\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            standardization_table
            .to_string(index=False)
        )

        output_file.write("\n\n")

        output_file.write(
            "RÉSUMÉ STATSMODELS\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            result.summary().as_text()
        )

        output_file.write("\n")


# ============================================================================
# NETTOYAGE DES ANCIENNES SORTIES
# ============================================================================

def remove_obsolete_output_files():
    """
    Supprime les anciennes sorties et réinitialise les fichiers actuels.
    """
    filenames = [
        # Anciennes sorties
        "below_ceiling_fixed_effects_ML.csv",
        "below_ceiling_fixed_effects_REML.csv",
        "below_ceiling_model_ML.txt",
        "below_ceiling_model_REML.txt",

        # Sorties actuelles à réinitialiser
        "ceiling_summary.csv",
        "below_ceiling_fixed_effects.csv",
        "below_ceiling_REML_summary.txt",
    ]

    for filename in filenames:
        path = (
            OUTPUT_DIR
            / filename
        )

        if path.is_file():
            path.unlink()

            print(
                "Ancienne sortie supprimée :",
                path,
            )


# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

def main():
    section(
        "ANALYSE DE SENSIBILITÉ AU PLAFOND — EXPÉRIENCE E1"
    )

    remove_obsolete_output_files()

    try:
        # ==================================================================
        # 1. Chargement et transformations sur le dataset complet
        # ==================================================================

        (
            full_data,
            sequence_mean,
            standardization_table,
        ) = load_and_prepare_data()

        # ==================================================================
        # 2. Exclusion des réponses égales à 100
        # ==================================================================

        below_ceiling_data = full_data.loc[
            full_data["confidence"] < 100
        ].copy()

        if below_ceiling_data.empty:
            raise ValueError(
                "Aucune observation ne reste après exclusion "
                "des réponses égales à 100."
            )

        if (
            below_ceiling_data[
                "subject_id"
            ].nunique()
            < 2
        ):
            raise ValueError(
                "Moins de deux participants restent sous le plafond."
            )

        if (
            below_ceiling_data[
                "item_id"
            ].nunique()
            < 2
        ):
            raise ValueError(
                "Moins de deux items restent sous le plafond."
            )

        # ==================================================================
        # 3. Résumé du plafond
        # ==================================================================

        ceiling_summary = (
            create_ceiling_summary(
                full_data=
                    full_data,
                below_ceiling_data=
                    below_ceiling_data,
            )
        )

        ceiling_summary.to_csv(
            CEILING_SUMMARY_FILE,
            index=False,
        )

        section(
            "DISTRIBUTION DU PLAFOND"
        )

        print(
            ceiling_summary
            .to_string(index=False)
        )

        # ==================================================================
        # 4. Ajustement REML sous le plafond
        # ==================================================================

        (
            result_reml,
            optimizer_reml,
        ) = fit_reml_model(
            below_ceiling_data
        )

        # ==================================================================
        # 5. Effets fixes
        # ==================================================================

        fixed_effects = (
            create_fixed_effects_table(
                result_reml
            )
        )

        fixed_effects.to_csv(
            FIXED_EFFECTS_FILE,
            index=False,
        )

        # ==================================================================
        # 6. Résumé complet
        # ==================================================================

        save_reml_summary(
            result=result_reml,
            optimizer=optimizer_reml,
            full_data=full_data,
            below_ceiling_data=
                below_ceiling_data,
            sequence_mean=sequence_mean,
            standardization_table=
                standardization_table,
        )

        # ==================================================================
        # 7. Affichage
        # ==================================================================

        section(
            "EFFETS FIXES SOUS LE PLAFOND — REML"
        )

        print(
            fixed_effects
            .to_string(index=False)
        )

        section(
            "RÉSUMÉ STATSMODELS"
        )

        print(
            result_reml.summary()
        )

        section(
            "FICHIERS PRODUITS"
        )

        for output_file in [
            CEILING_SUMMARY_FILE,
            FIXED_EFFECTS_FILE,
            REML_SUMMARY_FILE,
        ]:
            print(
                output_file
            )

        print("")
        print("=" * 80)
        print("ANALYSE SOUS LE PLAFOND TERMINÉE")
        print("=" * 80)

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        pd.errors.ParserError,
        pd.errors.MergeError,
    ) as error:
        section(
            "ERREUR"
        )

        print(
            type(error).__name__,
            ":",
            error,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()