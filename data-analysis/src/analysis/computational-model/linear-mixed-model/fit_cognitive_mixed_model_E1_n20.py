#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_cognitive_mixed_model_E1_n20.py

Ajuste le modèle linéaire mixte cognitif principal de la confiance
pour l'expérience E1.

Fichier d'entrée
----------------
dataset_analysis_E1_n20.csv

Modèles ajustés en ML
---------------------
1. Modèle nul :

    confidence ~ 1

2. Modèle de contrôle :

    confidence ~ condition + sequence_c10

3. Modèle cognitif :

    confidence ~
        condition
        + sequence_c10
        + subject_accuracy_z
        + item_entropy_z
        + subject_mean_models_z
        + models_within_subject_z

Le modèle cognitif est également ajusté en REML pour obtenir les
coefficients finaux et les composantes de variance.

La validité est encore incluse dans le modèle cognitif principal.
L'analyse de sensibilité déterminera ensuite si elle apporte une
amélioration suffisante pour être conservée dans le modèle final.

Effets aléatoires croisés
-------------------------
- intercept participant ;
- intercept item.

Fichiers produits
-----------------
cognitive_mixed_model_E1_n20/
    cognitive_model_REML_summary.txt
    cognitive_model_fixed_effects.csv
    model_fit_statistics.csv
    likelihood_ratio_tests.csv
    cognitive_model_metrics.csv
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
    / "cognitive_mixed_model_E1_n20"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------------------------------
# Fichiers produits
# --------------------------------------------------------------------------

REML_SUMMARY_FILE = (
    OUTPUT_DIR
    / "cognitive_model_REML_summary.txt"
)

FIXED_EFFECTS_FILE = (
    OUTPUT_DIR
    / "cognitive_model_fixed_effects.csv"
)

MODEL_FIT_STATISTICS_FILE = (
    OUTPUT_DIR
    / "model_fit_statistics.csv"
)

LIKELIHOOD_RATIO_TESTS_FILE = (
    OUTPUT_DIR
    / "likelihood_ratio_tests.csv"
)

MODEL_METRICS_FILE = (
    OUTPUT_DIR
    / "cognitive_model_metrics.csv"
)


# --------------------------------------------------------------------------
# Formules
# --------------------------------------------------------------------------

NULL_FORMULA = (
    "confidence ~ 1"
)

CONTROL_FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10"
)

COGNITIVE_FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z "
    "+ validity_binary"
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

ESSENTIAL_COLUMNS = [
    "confidence",
    "subject_id",
    "item_id",
    "condition",
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
    Convertit une valeur numérique en float.

    Retourne NaN si la valeur ne peut pas être convertie ou si elle
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


def standardize(series):
    """
    Standardise une variable avec l'écart-type empirique ddof=1.

    z = (x - moyenne) / écart-type
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
    Charge le dataset, vérifie les colonnes et construit les variables
    centrées ou standardisées.
    """
    section(
        "CHARGEMENT ET PRÉPARATION DES DONNÉES"
    )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE
    )

    print(
        "Fichier :",
        DATA_FILE,
    )

    print(
        "Nombre de lignes brutes :",
        len(data),
    )

    missing_columns = [
        column
        for column in ESSENTIAL_COLUMNS
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
    # Conversion des colonnes numériques
    # ------------------------------------------------------------------

    numeric_columns = [
        "confidence",
        "sequence",
        "subject_accuracy",
        "item_entropy",
        "subject_mean_models",
        "models_within_subject",
        "validity_binary",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

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
        .astype("string")
        .str.strip()
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
        subset=ESSENTIAL_COLUMNS
    ).copy()

    print(
        "Lignes supprimées pour donnée essentielle manquante :",
        before_drop - len(data),
    )

    # ------------------------------------------------------------------
    # Contrôles
    # ------------------------------------------------------------------

    invalid_confidence = (
        (data["confidence"] < 0)
        | (data["confidence"] > 100)
    )

    if invalid_confidence.any():
        raise ValueError(
            "Certaines confiances sont hors de [0, 100]."
        )

    valid_conditions = {
        "Neutral",
        "Standard",
    }

    observed_conditions = set(
        data["condition"]
        .dropna()
        .unique()
    )

    if not observed_conditions.issubset(
        valid_conditions
    ):
        raise ValueError(
            "Conditions inattendues : "
            f"{sorted(observed_conditions)}"
        )

    if observed_conditions != valid_conditions:
        raise ValueError(
            "Les conditions Neutral et Standard doivent "
            "toutes les deux être présentes."
        )

    validity_values = set(
        data["validity_binary"]
        .dropna()
        .unique()
    )

    if not validity_values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            "validity_binary doit uniquement contenir "
            "les valeurs 0 et 1. "
            f"Valeurs observées : {sorted(validity_values)}"
        )

    data["validity_binary"] = (
        data["validity_binary"]
        .astype(int)
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
    # Standardisation des prédicteurs cognitifs
    # ------------------------------------------------------------------

    standardization_information = []

    for column in STANDARDIZED_COLUMNS:
        standardized_column = (
            f"{column}_z"
        )

        (
            data[standardized_column],
            mean,
            standard_deviation,
        ) = standardize(
            data[column]
        )

        standardization_information.append({
            "variable":
                column,

            "standardized_variable":
                standardized_column,

            "mean":
                mean,

            "standard_deviation":
                standard_deviation,
        })

    standardization_table = pd.DataFrame(
        standardization_information
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
        "Moyenne de confiance :",
        round(
            data["confidence"].mean(),
            6,
        ),
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
# CONSTRUCTION DU MODÈLE
# ============================================================================

def build_model(
    data,
    formula,
):
    """
    Construit un modèle à intercepts aléatoires croisés.

    Toutes les observations sont placées dans un groupe artificiel
    unique. Les participants et les items sont introduits comme
    composantes de variance.
    """
    model_data = data.copy()

    model_data["_global_group"] = (
        "all_observations"
    )

    return smf.mixedlm(
        formula=formula,
        data=model_data,
        groups=model_data[
            "_global_group"
        ],
        re_formula="0",
        vc_formula=VC_FORMULA,
        use_sparse=USE_SPARSE_MATRICES,
    )


# ============================================================================
# AJUSTEMENT
# ============================================================================

def fit_model(
    data,
    formula,
    reml,
    model_name,
):
    """
    Essaie plusieurs optimiseurs.

    Le premier ajustement signalé comme convergé est conservé.
    """
    estimation = (
        "REML"
        if reml
        else "ML"
    )

    section(
        f"AJUSTEMENT — {model_name} — {estimation}"
    )

    print(
        "Formule :",
        formula,
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
            model = build_model(
                data=data,
                formula=formula,
            )

            with warnings.catch_warnings(
                record=True
            ) as caught_warnings:
                warnings.simplefilter(
                    "always"
                )

                result = model.fit(
                    reml=reml,
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
            f"{model_name} : un résultat a été obtenu, "
            "mais aucun optimiseur n'a signalé une "
            "convergence complète."
        )

    raise RuntimeError(
        f"Échec de tous les optimiseurs pour {model_name}. "
        f"Dernière erreur : {last_error!r}"
    )


# ============================================================================
# EXTRACTION DES EFFETS FIXES
# ============================================================================

def create_fixed_effects_table(result):
    """
    Extrait les coefficients REML et calcule les tests de Wald
    bilatéraux.

    Pour chaque coefficient :

        z = estimation / erreur standard

        p = 2 × P(Z > |z|)

    sous l'approximation :

        Z ~ N(0, 1)

    lorsque l'hypothèse nulle beta = 0 est vraie.
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

    ci_95_lower = (
        estimates
        - 1.96
        * standard_errors
    )

    ci_95_upper = (
        estimates
        + 1.96
        * standard_errors
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
            ci_95_lower,

        "ci_95_upper":
            ci_95_upper,
    })


# ============================================================================
# STATISTIQUES DES MODÈLES
# ============================================================================

def count_estimated_parameters(result):
    """
    Retourne le nombre de paramètres rapporté par le modèle.

    Pour les différences de degrés de liberté du test LR, seuls les
    paramètres ajoutés entre les deux modèles sont importants.
    """
    return int(
        len(result.params)
    )


def create_model_statistics(
    result,
    model_name,
    formula,
    estimation,
    optimizer,
):
    """
    Crée une ligne de statistiques générales pour un modèle.
    """
    return {
        "model":
            model_name,

        "estimation":
            estimation,

        "formula":
            formula,

        "converged":
            bool(
                getattr(
                    result,
                    "converged",
                    False,
                )
            ),

        "optimizer":
            optimizer,

        "log_likelihood":
            safe_float(
                result.llf
            ),

        # AIC et BIC sont utilisés pour les modèles ML.
        "aic":
            (
                safe_float(result.aic)
                if estimation == "ML"
                else np.nan
            ),

        "bic":
            (
                safe_float(result.bic)
                if estimation == "ML"
                else np.nan
            ),

        "number_of_estimated_parameters":
            count_estimated_parameters(
                result
            ),

        "n_observations":
            int(
                result.nobs
            ),

        "residual_variance":
            safe_float(
                result.scale
            ),
    }


# ============================================================================
# TESTS DU RAPPORT DE VRAISEMBLANCE
# ============================================================================

def likelihood_ratio_test(
    smaller_result,
    larger_result,
):
    """
    Compare deux modèles emboîtés ajustés en ML.

    LR = 2 × (logL_grand - logL_petit)

    Sous l'hypothèse nulle selon laquelle les paramètres ajoutés
    n'améliorent pas le modèle :

        LR suit approximativement une loi du chi-deux.

    Les degrés de liberté correspondent au nombre de paramètres
    supplémentaires.
    """
    smaller_log_likelihood = safe_float(
        smaller_result.llf
    )

    larger_log_likelihood = safe_float(
        larger_result.llf
    )

    likelihood_ratio = (
        2
        * (
            larger_log_likelihood
            - smaller_log_likelihood
        )
    )

    degrees_of_freedom = (
        count_estimated_parameters(
            larger_result
        )
        - count_estimated_parameters(
            smaller_result
        )
    )

    if degrees_of_freedom <= 0:
        p_value = np.nan

    else:
        p_value = stats.chi2.sf(
            likelihood_ratio,
            degrees_of_freedom,
        )

    return {
        "likelihood_ratio":
            likelihood_ratio,

        "degrees_of_freedom":
            degrees_of_freedom,

        "p_value":
            p_value,
    }


# ============================================================================
# VARIANCES ET R²
# ============================================================================

def get_variance_components(result):
    """
    Extrait les variances participant, item et résiduelle.
    """
    try:
        component_names = list(
            result.model.exog_vc.names
        )

    except (AttributeError, TypeError):
        component_names = []

    component_values = np.asarray(
        result.vcomp,
        dtype=float,
    )

    if not component_names:
        component_names = [
            f"component_{index}"
            for index in range(
                len(component_values)
            )
        ]

    variance_map = {
        str(name).strip().lower():
            float(value)

        for name, value in zip(
            component_names,
            component_values,
        )
    }

    participant_variance = np.nan
    item_variance = np.nan

    for name, variance in variance_map.items():
        if "subject" in name:
            participant_variance = variance

        elif "item" in name:
            item_variance = variance

    if (
        pd.isna(participant_variance)
        or pd.isna(item_variance)
    ):
        raise RuntimeError(
            "Impossible d'identifier les variances "
            "participant et item. "
            f"Composantes trouvées : {variance_map}"
        )

    return {
        "participant_variance":
            participant_variance,

        "item_variance":
            item_variance,

        "residual_variance":
            float(result.scale),
    }


def calculate_model_metrics(result):
    """
    Calcule les composantes de variance et les R² de Nakagawa.

    R² marginal :
        variance expliquée par les effets fixes.

    R² conditionnel :
        variance expliquée par les effets fixes et aléatoires.
    """
    variances = get_variance_components(
        result
    )

    fixed_predictions = np.asarray(
        result.model.exog
        @ np.asarray(
            result.fe_params
        ),
        dtype=float,
    )

    fixed_effect_variance = float(
        np.var(
            fixed_predictions,
            ddof=1,
        )
    )

    participant_variance = (
        variances[
            "participant_variance"
        ]
    )

    item_variance = (
        variances[
            "item_variance"
        ]
    )

    residual_variance = (
        variances[
            "residual_variance"
        ]
    )

    total_variance = (
        fixed_effect_variance
        + participant_variance
        + item_variance
        + residual_variance
    )

    marginal_r2 = (
        fixed_effect_variance
        / total_variance
    )

    conditional_r2 = (
        fixed_effect_variance
        + participant_variance
        + item_variance
    ) / total_variance

    random_total = (
        participant_variance
        + item_variance
        + residual_variance
    )

    return {
        "fixed_effect_variance":
            fixed_effect_variance,

        "participant_variance":
            participant_variance,

        "participant_standard_deviation":
            np.sqrt(
                participant_variance
            ),

        "participant_proportion_random_total":
            (
                participant_variance
                / random_total
            ),

        "item_variance":
            item_variance,

        "item_standard_deviation":
            np.sqrt(
                item_variance
            ),

        "item_proportion_random_total":
            (
                item_variance
                / random_total
            ),

        "residual_variance":
            residual_variance,

        "residual_standard_deviation":
            np.sqrt(
                residual_variance
            ),

        "residual_proportion_random_total":
            (
                residual_variance
                / random_total
            ),

        "total_variance":
            total_variance,

        "marginal_r2":
            marginal_r2,

        "conditional_r2":
            conditional_r2,
    }


# ============================================================================
# SAUVEGARDE DU RÉSUMÉ
# ============================================================================

def save_reml_summary(
    result,
    optimizer,
    sequence_mean,
    standardization_table,
    metrics,
):
    """
    Sauvegarde le résumé complet du modèle cognitif final.
    """
    with open(
        REML_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "MODÈLE MIXTE COGNITIF PRINCIPAL E1 — REML\n"
        )

        output_file.write(
            "=" * 80
        )

        output_file.write("\n\n")

        output_file.write(
            "FORMULE\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        output_file.write(
            COGNITIVE_FORMULA
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
            "STANDARDISATION DES PRÉDICTEURS\n"
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

        output_file.write("\n\n")

        output_file.write(
            "VARIANCES ET R²\n"
        )

        output_file.write(
            "-" * 80
        )

        output_file.write("\n")

        for name, value in metrics.items():
            output_file.write(
                f"{name}: {value:.10f}\n"
            )


# ============================================================================
# NETTOYAGE DES ANCIENNES SORTIES
# ============================================================================

def remove_obsolete_output_files():
    """
    Supprime les anciennes sorties que la version simplifiée
    ne produit plus.
    """
    obsolete_filenames = [
        "predictor_standardization.csv",
        "cognitive_predictor_correlations.csv",
        "high_predictor_correlations.csv",
        "cognitive_model_ML_summary.txt",
        "cognitive_model_fixed_effects_ML.csv",
        "cognitive_model_fixed_effects_REML.csv",
        "model_comparison.csv",
        "variance_components.csv",
        "model_r2.csv",
        "cognitive_model_predictions.csv",
        "cognitive_model_results.json",
    ]

    for filename in obsolete_filenames:
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
        "MODÈLE MIXTE COGNITIF PRINCIPAL — EXPÉRIENCE E1"
    )

    remove_obsolete_output_files()

    try:
        # ==================================================================
        # 1. Données
        # ==================================================================

        (
            data,
            sequence_mean,
            standardization_table,
        ) = load_and_prepare_data()

        # ==================================================================
        # 2. Modèles ML
        # ==================================================================

        (
            null_ml,
            null_optimizer,
        ) = fit_model(
            data=data,
            formula=NULL_FORMULA,
            reml=False,
            model_name="Modèle nul",
        )

        (
            control_ml,
            control_optimizer,
        ) = fit_model(
            data=data,
            formula=CONTROL_FORMULA,
            reml=False,
            model_name="Modèle de contrôle",
        )

        (
            cognitive_ml,
            cognitive_ml_optimizer,
        ) = fit_model(
            data=data,
            formula=COGNITIVE_FORMULA,
            reml=False,
            model_name="Modèle cognitif",
        )

        # ==================================================================
        # 3. Modèle final REML
        # ==================================================================

        (
            cognitive_reml,
            cognitive_reml_optimizer,
        ) = fit_model(
            data=data,
            formula=COGNITIVE_FORMULA,
            reml=True,
            model_name="Modèle cognitif",
        )

        # ==================================================================
        # 4. Coefficients REML
        # ==================================================================

        fixed_effects = (
            create_fixed_effects_table(
                cognitive_reml
            )
        )

        fixed_effects.to_csv(
            FIXED_EFFECTS_FILE,
            index=False,
        )

        # ==================================================================
        # 5. Statistiques d'ajustement
        # ==================================================================

        fit_rows = [
            create_model_statistics(
                result=null_ml,
                model_name="Null",
                formula=NULL_FORMULA,
                estimation="ML",
                optimizer=null_optimizer,
            ),
            create_model_statistics(
                result=control_ml,
                model_name="Control",
                formula=CONTROL_FORMULA,
                estimation="ML",
                optimizer=control_optimizer,
            ),
            create_model_statistics(
                result=cognitive_ml,
                model_name="Cognitive",
                formula=COGNITIVE_FORMULA,
                estimation="ML",
                optimizer=
                    cognitive_ml_optimizer,
            ),
            create_model_statistics(
                result=cognitive_reml,
                model_name="Cognitive",
                formula=COGNITIVE_FORMULA,
                estimation="REML",
                optimizer=
                    cognitive_reml_optimizer,
            ),
        ]

        fit_statistics = pd.DataFrame(
            fit_rows
        )

        fit_statistics.to_csv(
            MODEL_FIT_STATISTICS_FILE,
            index=False,
        )

        # ==================================================================
        # 6. Tests du rapport de vraisemblance
        # ==================================================================

        null_vs_control = (
            likelihood_ratio_test(
                smaller_result=null_ml,
                larger_result=control_ml,
            )
        )

        control_vs_cognitive = (
            likelihood_ratio_test(
                smaller_result=control_ml,
                larger_result=cognitive_ml,
            )
        )

        null_vs_cognitive = (
            likelihood_ratio_test(
                smaller_result=null_ml,
                larger_result=cognitive_ml,
            )
        )

        likelihood_ratio_table = pd.DataFrame([
            {
                "comparison":
                    "Null vs Control",

                **null_vs_control,
            },
            {
                "comparison":
                    "Control vs Cognitive",

                **control_vs_cognitive,
            },
            {
                "comparison":
                    "Null vs Cognitive",

                **null_vs_cognitive,
            },
        ])

        likelihood_ratio_table.to_csv(
            LIKELIHOOD_RATIO_TESTS_FILE,
            index=False,
        )

        # ==================================================================
        # 7. Variances et R²
        # ==================================================================

        metrics = calculate_model_metrics(
            cognitive_reml
        )

        metrics_table = pd.DataFrame([
            {
                "model":
                    "Cognitive_REML",

                **metrics,
            }
        ])

        metrics_table.to_csv(
            MODEL_METRICS_FILE,
            index=False,
        )

        # ==================================================================
        # 8. Résumé complet
        # ==================================================================

        save_reml_summary(
            result=cognitive_reml,
            optimizer=
                cognitive_reml_optimizer,
            sequence_mean=
                sequence_mean,
            standardization_table=
                standardization_table,
            metrics=metrics,
        )

        # ==================================================================
        # 9. Affichage
        # ==================================================================

        section(
            "COEFFICIENTS FIXES — REML"
        )

        print(
            fixed_effects
            .to_string(index=False)
        )

        section(
            "COMPARAISON DES MODÈLES"
        )

        print(
            fit_statistics
            .to_string(index=False)
        )

        section(
            "TESTS DU RAPPORT DE VRAISEMBLANCE"
        )

        print(
            likelihood_ratio_table
            .to_string(index=False)
        )

        section(
            "VARIANCES ET R² DU MODÈLE FINAL"
        )

        print(
            metrics_table
            .to_string(index=False)
        )

        section(
            "FICHIERS PRODUITS"
        )

        for output_file in [
            REML_SUMMARY_FILE,
            FIXED_EFFECTS_FILE,
            MODEL_FIT_STATISTICS_FILE,
            LIKELIHOOD_RATIO_TESTS_FILE,
            MODEL_METRICS_FILE,
        ]:
            print(
                output_file
            )

        print("")
        print("=" * 80)
        print("MODÈLE COGNITIF TERMINÉ")
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
