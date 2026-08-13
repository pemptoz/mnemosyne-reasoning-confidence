#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_ceiling_logistic_mixed_model_E1.py

Ajuste un modèle logistique mixte bayésien approché prédisant
l'utilisation de la valeur maximale de confiance dans l'expérience E1.

Fichier d'entrée
----------------
results/tables/computational_model/dataset_analysis_E1_n20.csv

Variable dépendante
-------------------
at_ceiling = 1 si confidence == 100
at_ceiling = 0 si confidence < 100

Modèle
------
at_ceiling ~
    condition
    + sequence_c10
    + subject_accuracy_z
    + item_entropy_z
    + subject_mean_models_z
    + models_within_subject_z

Effets aléatoires croisés
-------------------------
- intercept participant ;
- intercept item.

Estimation
----------
Approximation variationnelle bayésienne avec :

    statsmodels.BinomialBayesMixedGLM.fit_vb()

Les coefficients sont exprimés en log-odds.

Les odds ratios sont obtenus par :

    OR = exp(coefficient)

Fichiers produits
-----------------
ceiling_logistic_mixed_model_E1_n20/
    ceiling_by_condition.csv
    ceiling_logistic_fixed_effects.csv
    adjusted_ceiling_probabilities.csv
    ceiling_logistic_model_summary.txt
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit

from statsmodels.genmod.bayes_mixed_glm import (
    BinomialBayesMixedGLM,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "computational_model"
    / "dataset_analysis_E1_n20.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "computational_model"
    / "ceiling_logistic_mixed_model_E1_n20"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------
# Fichiers produits
# --------------------------------------------------------------------------

CONDITION_SUMMARY_FILE = (
    OUTPUT_DIR
    / "ceiling_by_condition.csv"
)

FIXED_EFFECTS_FILE = (
    OUTPUT_DIR
    / "ceiling_logistic_fixed_effects.csv"
)

ADJUSTED_PROBABILITIES_FILE = (
    OUTPUT_DIR
    / "adjusted_ceiling_probabilities.csv"
)

MODEL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "ceiling_logistic_model_summary.txt"
)


# --------------------------------------------------------------------------
# Formule
# --------------------------------------------------------------------------

FORMULA = (
    "at_ceiling ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z"
)


# --------------------------------------------------------------------------
# Effets aléatoires
# --------------------------------------------------------------------------

VC_FORMULAS = {
    "participant":
        "0 + C(subject_id)",

    "item":
        "0 + C(item_id)",
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

NUMERIC_COLUMNS = [
    "confidence",
    "sequence",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]

STANDARDIZED_VARIABLES = [
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]


# --------------------------------------------------------------------------
# A priori
# --------------------------------------------------------------------------

# Écart-type a priori des coefficients fixes.
FE_P = 2.0

# Écart-type a priori des logarithmes des écarts-types aléatoires.
VCP_P = 0.5


# --------------------------------------------------------------------------
# Tentatives d'optimisation
# --------------------------------------------------------------------------

OPTIMIZATION_ATTEMPTS = [
    {
        "name":
            "BFGS_scale_fe_true_gtol_1e-5",

        "fit_method":
            "BFGS",

        "scale_fe":
            True,

        "minim_opts": {
            "maxiter": 10000,
            "gtol": 1e-5,
        },
    },
    {
        "name":
            "BFGS_scale_fe_true_gtol_1e-4",

        "fit_method":
            "BFGS",

        "scale_fe":
            True,

        "minim_opts": {
            "maxiter": 10000,
            "gtol": 1e-4,
        },
    },
    {
        "name":
            "L-BFGS-B_scale_fe_true",

        "fit_method":
            "L-BFGS-B",

        "scale_fe":
            True,

        "minim_opts": {
            "maxiter": 15000,
            "ftol": 1e-10,
            "gtol": 1e-5,
            "maxls": 50,
        },
    },
]


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
    Charge et prépare le dataset pour le modèle logistique.
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
    # Conversion des variables numériques
    # ------------------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------
    # Normalisation
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
    # Suppression des valeurs essentielles manquantes
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
    # Variable dépendante binaire
    # ------------------------------------------------------------------

    calculated_ceiling = (
        data["confidence"] == 100
    ).astype(int)

    if "is_ceiling" in data.columns:
        provided_ceiling = pd.to_numeric(
            data["is_ceiling"],
            errors="coerce",
        )

        inconsistent_ceiling = (
            provided_ceiling.notna()
            & (
                provided_ceiling
                != calculated_ceiling
            )
        )

        if inconsistent_ceiling.any():
            raise ValueError(
                "La colonne is_ceiling n'est pas cohérente "
                "avec confidence == 100."
            )

    data["at_ceiling"] = (
        calculated_ceiling
    )

    if set(
        data["at_ceiling"].unique()
    ) != {0, 1}:
        raise ValueError(
            "at_ceiling doit contenir les deux modalités 0 et 1."
        )

    # ------------------------------------------------------------------
    # Centrage de la séquence
    # ------------------------------------------------------------------

    sequence_mean = float(
        data["sequence"].mean()
    )

    data["sequence_c10"] = (
        data["sequence"]
        - sequence_mean
    ) / 10.0

    # ------------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------------

    standardization_rows = []

    for variable in STANDARDIZED_VARIABLES:
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
        "Nombre de lignes utilisées :",
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

    print(
        "Nombre de réponses égales à 100 :",
        int(
            data["at_ceiling"].sum()
        ),
    )

    print(
        "Taux global au plafond :",
        float(
            data["at_ceiling"].mean()
        ),
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
# DESCRIPTIF PAR CONDITION
# ============================================================================

def create_condition_summary(data):
    """
    Calcule les taux bruts de réponses égales à 100 dans chaque
    condition.
    """
    return (
        data
        .groupby(
            "condition",
            as_index=False,
        )
        .agg(
            n_observations=(
                "at_ceiling",
                "size",
            ),

            n_at_ceiling=(
                "at_ceiling",
                "sum",
            ),

            ceiling_rate=(
                "at_ceiling",
                "mean",
            ),

            mean_confidence=(
                "confidence",
                "mean",
            ),
        )
    )


# ============================================================================
# CONSTRUCTION DU MODÈLE
# ============================================================================

def build_model(data):
    """
    Construit le modèle logistique mixte bayésien.
    """
    section(
        "CONSTRUCTION DU MODÈLE"
    )

    print(
        "Formule fixe :",
        FORMULA,
    )

    print("")
    print(
        "Composantes aléatoires :"
    )

    for name, formula in (
        VC_FORMULAS.items()
    ):
        print(
            f"  {name} : {formula}"
        )

    print("")
    print(
        "A priori fe_p :",
        FE_P,
    )

    print(
        "A priori vcp_p :",
        VCP_P,
    )

    return BinomialBayesMixedGLM.from_formula(
        formula=FORMULA,
        vc_formulas=VC_FORMULAS,
        data=data,
        fe_p=FE_P,
        vcp_p=VCP_P,
    )


# ============================================================================
# AJUSTEMENT VARIATIONNEL
# ============================================================================

def fit_variational_model(model):
    """
    Essaie plusieurs configurations d'optimisation.

    Le premier ajustement ayant officiellement convergé est retenu.
    """
    section(
        "AJUSTEMENT VARIATIONNEL BAYÉSIEN"
    )

    attempt_rows = []

    for (
        attempt_number,
        attempt,
    ) in enumerate(
        OPTIMIZATION_ATTEMPTS,
        start=1,
    ):
        print("")
        print("-" * 80)

        print(
            f"TENTATIVE {attempt_number}/"
            f"{len(OPTIMIZATION_ATTEMPTS)} : "
            f"{attempt['name']}"
        )

        print("-" * 80)

        try:
            with warnings.catch_warnings(
                record=True
            ) as caught_warnings:
                warnings.simplefilter(
                    "always"
                )

                candidate = model.fit_vb(
                    fit_method=
                        attempt["fit_method"],
                    minim_opts=
                        attempt["minim_opts"],
                    scale_fe=
                        attempt["scale_fe"],
                    verbose=False,
                )

            optimization_results = (
                getattr(
                    candidate,
                    "optim_retvals",
                    {},
                )
                or {}
            )

            success = bool(
                optimization_results.get(
                    "success",
                    False,
                )
            )

            objective = safe_float(
                optimization_results.get(
                    "fun",
                    np.nan,
                )
            )

            iterations = (
                optimization_results.get(
                    "nit",
                    np.nan,
                )
            )

            message = str(
                optimization_results.get(
                    "message",
                    "",
                )
            )

            gradient = (
                optimization_results.get(
                    "jac",
                    None,
                )
            )

            if gradient is None:
                maximum_absolute_gradient = (
                    np.nan
                )

            else:
                gradient = np.asarray(
                    gradient,
                    dtype=float,
                )

                maximum_absolute_gradient = float(
                    np.max(
                        np.abs(
                            gradient
                        )
                    )
                )

            warning_messages = [
                str(warning.message)
                for warning in caught_warnings
            ]

            attempt_rows.append({
                "attempt":
                    attempt["name"],

                "success":
                    success,

                "objective":
                    objective,

                "iterations":
                    iterations,

                "maximum_absolute_gradient":
                    maximum_absolute_gradient,

                "message":
                    message,

                "warnings":
                    " | ".join(
                        warning_messages
                    ),
            })

            print(
                "Succès :",
                success,
            )

            print(
                "Objectif :",
                objective,
            )

            print(
                "Itérations :",
                iterations,
            )

            print(
                "Gradient absolu maximal :",
                maximum_absolute_gradient,
            )

            print(
                "Message :",
                message,
            )

            if warning_messages:
                print(
                    "Avertissements :"
                )

                for warning_message in (
                    warning_messages
                ):
                    print(
                        "  -",
                        warning_message,
                    )

            if success:
                attempt_table = pd.DataFrame(
                    attempt_rows
                )

                return (
                    candidate,
                    attempt["name"],
                    optimization_results,
                    attempt_table,
                )

        except Exception as error:
            attempt_rows.append({
                "attempt":
                    attempt["name"],

                "success":
                    False,

                "objective":
                    np.nan,

                "iterations":
                    np.nan,

                "maximum_absolute_gradient":
                    np.nan,

                "message":
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),

                "warnings":
                    "",
            })

            print(
                "Échec :",
                repr(error),
            )

    attempt_table = pd.DataFrame(
        attempt_rows
    )

    raise RuntimeError(
        "Aucune tentative d'optimisation n'a officiellement "
        "convergé.\n\n"
        + attempt_table.to_string(
            index=False
        )
    )


# ============================================================================
# EFFETS FIXES
# ============================================================================

def create_fixed_effects_table(result):
    """
    Extrait les effets fixes postérieurs.

    Les intervalles crédibles sont calculés par l'approximation :

        moyenne postérieure ± 1,96 × écart-type postérieur

    Les odds ratios sont calculés par :

        OR = exp(coefficient)
    """
    parameter_names = list(
        result.model.fep_names
    )

    posterior_means = np.asarray(
        result.fe_mean,
        dtype=float,
    )

    posterior_standard_deviations = (
        np.asarray(
            result.fe_sd,
            dtype=float,
        )
    )

    credible_lower = (
        posterior_means
        - 1.96
        * posterior_standard_deviations
    )

    credible_upper = (
        posterior_means
        + 1.96
        * posterior_standard_deviations
    )

    odds_ratios = np.exp(
        posterior_means
    )

    odds_ratio_lower = np.exp(
        credible_lower
    )

    odds_ratio_upper = np.exp(
        credible_upper
    )

    return pd.DataFrame({
        "parameter":
            parameter_names,

        "posterior_mean_log_odds":
            posterior_means,

        "posterior_sd":
            posterior_standard_deviations,

        "credible_95_lower_log_odds":
            credible_lower,

        "credible_95_upper_log_odds":
            credible_upper,

        "odds_ratio":
            odds_ratios,

        "credible_95_lower_odds_ratio":
            odds_ratio_lower,

        "credible_95_upper_odds_ratio":
            odds_ratio_upper,
    })


# ============================================================================
# EFFETS ALÉATOIRES
# ============================================================================

def create_random_effects_table(result):
    """
    Extrait les écarts-types des deux familles d'effets aléatoires.

    Les paramètres vcp sont exprimés comme le logarithme de
    l'écart-type :

        log(SD)

    L'écart-type est donc récupéré par :

        SD = exp(log(SD))
    """
    component_names = list(
        result.model.vcp_names
    )

    posterior_log_sd_means = np.asarray(
        result.vcp_mean,
        dtype=float,
    )

    posterior_log_sd_sds = np.asarray(
        result.vcp_sd,
        dtype=float,
    )

    credible_log_sd_lower = (
        posterior_log_sd_means
        - 1.96
        * posterior_log_sd_sds
    )

    credible_log_sd_upper = (
        posterior_log_sd_means
        + 1.96
        * posterior_log_sd_sds
    )

    random_standard_deviations = np.exp(
        posterior_log_sd_means
    )

    random_sd_lower = np.exp(
        credible_log_sd_lower
    )

    random_sd_upper = np.exp(
        credible_log_sd_upper
    )

    return pd.DataFrame({
        "component":
            component_names,

        "posterior_mean_log_sd":
            posterior_log_sd_means,

        "posterior_sd_log_sd":
            posterior_log_sd_sds,

        "random_effect_standard_deviation_logit":
            random_standard_deviations,

        "credible_95_lower_standard_deviation":
            random_sd_lower,

        "credible_95_upper_standard_deviation":
            random_sd_upper,

        "approximate_variance_logit":
            random_standard_deviations
            ** 2,
    })


# ============================================================================
# PRÉDICTIONS POUR DES SCÉNARIOS
# ============================================================================

def get_fixed_coefficient(
    fixed_effects,
    parameter_name,
):
    """
    Récupère un coefficient dans le tableau des effets fixes.
    """
    selected = fixed_effects.loc[
        fixed_effects["parameter"]
        == parameter_name,
        "posterior_mean_log_odds",
    ]

    if len(selected) != 1:
        raise KeyError(
            "Impossible d'identifier le coefficient "
            f"{parameter_name!r}."
        )

    return float(
        selected.iloc[0]
    )


def create_adjusted_scenario_predictions(
    fixed_effects,
    sequence_mean,
):
    """
    Calcule les probabilités ajustées d'utiliser 100 :

    - au premier essai ;
    - à la séquence moyenne ;
    - au dernier essai ;

    pour les conditions Neutral et Standard.

    Les quatre prédicteurs cognitifs standardisés sont fixés à zéro,
    c'est-à-dire à leur moyenne.
    """
    intercept = get_fixed_coefficient(
        fixed_effects,
        "Intercept",
    )

    condition_parameter = (
        "C(condition, Treatment(reference='Neutral'))"
        "[T.Standard]"
    )

    condition_effect = (
        get_fixed_coefficient(
            fixed_effects,
            condition_parameter,
        )
    )

    sequence_effect = (
        get_fixed_coefficient(
            fixed_effects,
            "sequence_c10",
        )
    )

    sequence_values = {
        "Beginning_sequence_1":
            (
                1.0
                - sequence_mean
            )
            / 10.0,

        "Middle_sequence_mean":
            0.0,

        "End_sequence_64":
            (
                64.0
                - sequence_mean
            )
            / 10.0,
    }

    rows = []

    for condition in [
        "Neutral",
        "Standard",
    ]:
        condition_offset = (
            condition_effect
            if condition == "Standard"
            else 0.0
        )

        for (
            position_name,
            sequence_c10,
        ) in sequence_values.items():
            predicted_log_odds = (
                intercept
                + condition_offset
                + sequence_effect
                * sequence_c10
            )

            predicted_probability = float(
                expit(
                    predicted_log_odds
                )
            )

            rows.append({
                "condition":
                    condition,

                "sequence_position":
                    position_name,

                "sequence_c10":
                    sequence_c10,

                "predicted_log_odds_fixed_only":
                    predicted_log_odds,

                "predicted_probability_fixed_only":
                    predicted_probability,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SAUVEGARDE DU RÉSUMÉ
# ============================================================================

def save_model_summary(
    result,
    selected_attempt,
    optimization_results,
    attempt_table,
    sequence_mean,
    standardization_table,
    condition_summary,
    random_effects,
):
    """
    Sauvegarde le résumé complet du modèle.
    """
    with open(
        MODEL_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "GLMM LOGISTIQUE DU PLAFOND DE CONFIANCE — E1\n"
        )

        output_file.write(
            "=" * 80
        )

        output_file.write("\n\n")

        output_file.write(
            "VARIABLE DÉPENDANTE\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            "at_ceiling = 1 si confidence == 100, "
            "0 sinon.\n\n"
        )

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
            "A PRIORI\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            f"fe_p = {FE_P}\n"
        )

        output_file.write(
            f"vcp_p = {VCP_P}\n\n"
        )

        output_file.write(
            "PRÉPARATION DES PRÉDICTEURS\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            f"Centre de la séquence : {sequence_mean:.6f}\n"
        )

        output_file.write(
            "Une unité de sequence_c10 correspond à dix essais.\n\n"
        )

        output_file.write(
            standardization_table
            .to_string(index=False)
        )

        output_file.write("\n\n")

        output_file.write(
            "TAUX BRUTS PAR CONDITION\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            condition_summary
            .to_string(index=False)
        )

        output_file.write("\n\n")

        output_file.write(
            "OPTIMISATION\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            f"Tentative retenue : {selected_attempt}\n"
        )

        output_file.write(
            "Succès : "
            f"{bool(optimization_results.get('success', False))}\n"
        )

        output_file.write(
            "Message : "
            f"{optimization_results.get('message', '')}\n"
        )

        output_file.write(
            "Itérations : "
            f"{optimization_results.get('nit', 'NA')}\n"
        )

        gradient = optimization_results.get(
            "jac",
            None,
        )

        if gradient is not None:
            gradient = np.asarray(
                gradient,
                dtype=float,
            )

            maximum_absolute_gradient = float(
                np.max(
                    np.abs(
                        gradient
                    )
                )
            )

            output_file.write(
                "Gradient absolu maximal : "
                f"{maximum_absolute_gradient:.10g}\n"
            )

        output_file.write("\n")

        output_file.write(
            "Toutes les tentatives réalisées :\n"
        )

        output_file.write(
            attempt_table
            .to_string(index=False)
        )

        output_file.write("\n\n")

        output_file.write(
            "EFFETS ALÉATOIRES\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            random_effects
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
    Supprime les anciennes sorties et réinitialise les sorties
    actuellement produites.
    """
    filenames = [
        # Anciennes sorties
        "predictor_standardization.csv",
        "ceiling_global_summary.csv",
        "ceiling_by_sequence_block.csv",
        "ceiling_logistic_optimization_attempts.csv",
        "ceiling_logistic_predictions.csv",
        "ceiling_logistic_calibration.csv",
        "ceiling_logistic_results.json",
        "ceiling_logistic_random_effect_standard_deviations.csv",

        # Sorties actuelles
        "ceiling_by_condition.csv",
        "ceiling_logistic_fixed_effects.csv",
        "adjusted_ceiling_probabilities.csv",
        "ceiling_logistic_model_summary.txt",
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
        "GLMM LOGISTIQUE DU PLAFOND DE CONFIANCE — E1"
    )

    remove_obsolete_output_files()

    try:
        # ==================================================================
        # 1. Chargement et préparation
        # ==================================================================

        (
            data,
            sequence_mean,
            standardization_table,
        ) = load_and_prepare_data()

        # ==================================================================
        # 2. Résumé brut par condition
        # ==================================================================

        condition_summary = (
            create_condition_summary(
                data
            )
        )

        condition_summary.to_csv(
            CONDITION_SUMMARY_FILE,
            index=False,
        )

        section(
            "TAUX BRUTS PAR CONDITION"
        )

        print(
            condition_summary
            .to_string(index=False)
        )

        # ==================================================================
        # 3. Construction du modèle
        # ==================================================================

        model = build_model(
            data
        )

        # ==================================================================
        # 4. Ajustement variationnel
        # ==================================================================

        (
            result,
            selected_attempt,
            optimization_results,
            attempt_table,
        ) = fit_variational_model(
            model
        )

        # Vérification supplémentaire.
        optimization_success = bool(
            optimization_results.get(
                "success",
                False,
            )
        )

        if not optimization_success:
            raise RuntimeError(
                "La tentative retenue n'a pas officiellement convergé."
            )

        # ==================================================================
        # 5. Effets fixes
        # ==================================================================

        fixed_effects = (
            create_fixed_effects_table(
                result
            )
        )

        fixed_effects.to_csv(
            FIXED_EFFECTS_FILE,
            index=False,
        )

        section(
            "EFFETS FIXES"
        )

        print(
            fixed_effects
            .to_string(index=False)
        )

        # ==================================================================
        # 6. Effets aléatoires
        # ==================================================================

        random_effects = (
            create_random_effects_table(
                result
            )
        )

        section(
            "EFFETS ALÉATOIRES"
        )

        print(
            random_effects
            .to_string(index=False)
        )

        # ==================================================================
        # 7. Scénarios ajustés
        # ==================================================================

        adjusted_probabilities = (
            create_adjusted_scenario_predictions(
                fixed_effects=
                    fixed_effects,
                sequence_mean=
                    sequence_mean,
            )
        )

        adjusted_probabilities.to_csv(
            ADJUSTED_PROBABILITIES_FILE,
            index=False,
        )

        section(
            "PROBABILITÉS AJUSTÉES"
        )

        print(
            adjusted_probabilities
            .to_string(index=False)
        )

        # ==================================================================
        # 8. Résumé complet
        # ==================================================================

        save_model_summary(
            result=result,
            selected_attempt=
                selected_attempt,
            optimization_results=
                optimization_results,
            attempt_table=
                attempt_table,
            sequence_mean=
                sequence_mean,
            standardization_table=
                standardization_table,
            condition_summary=
                condition_summary,
            random_effects=
                random_effects,
        )

        # ==================================================================
        # 9. Fichiers produits
        # ==================================================================

        section(
            "FICHIERS PRODUITS"
        )

        for output_file in [
            CONDITION_SUMMARY_FILE,
            FIXED_EFFECTS_FILE,
            ADJUSTED_PROBABILITIES_FILE,
            MODEL_SUMMARY_FILE,
        ]:
            print(
                output_file
            )

        print("")
        print("=" * 80)
        print("MODÈLE LOGISTIQUE DU PLAFOND TERMINÉ")
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
