#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_control_mixed_model_E1.py

Ajuste le modèle mixte de contrôle de la confiance pour
l'expérience E1.

Fichier d'entrée
----------------
dataset_analysis_E1_n20.csv

Modèle nul
----------
confidence ~ 1

Modèle de contrôle
------------------
confidence ~
    condition
    + sequence_c10

La condition Neutral est la catégorie de référence.

La variable sequence_c10 est définie par :

    sequence_c10 = (sequence - moyenne(sequence)) / 10

Son coefficient représente donc la variation moyenne de confiance
associée à dix essais supplémentaires.

Effets aléatoires croisés
-------------------------
- intercept participant ;
- intercept item.

Estimations
-----------
ML :
    utilisé pour comparer le modèle nul au modèle de contrôle.

REML :
    utilisé pour présenter les coefficients et les composantes de
    variance du modèle de contrôle.

Fichiers produits
-----------------
control_mixed_model_E1_n20/
    control_model_REML_summary.txt
    control_model_fixed_effects.csv
    model_comparison.csv
    control_model_metrics.csv
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
    / "control_mixed_model_E1_n20"
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
    / "control_model_REML_summary.txt"
)

FIXED_EFFECTS_FILE = (
    OUTPUT_DIR
    / "control_model_fixed_effects.csv"
)

MODEL_COMPARISON_FILE = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

MODEL_METRICS_FILE = (
    OUTPUT_DIR
    / "control_model_metrics.csv"
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
    "subject_id",
    "item_id",
    "condition",
    "sequence",
]


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------

REFERENCE_CONDITION = "Neutral"

SEQUENCE_SCALE = 10.0

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


# ============================================================================
# CHARGEMENT ET PRÉPARATION
# ============================================================================

def load_and_prepare_data():
    """
    Charge le dataset analytique et construit sequence_c10.
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
    # Filtrage des lignes déclarées complètes
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
    # Conversion et normalisation
    # ------------------------------------------------------------------

    data["confidence"] = pd.to_numeric(
        data["confidence"],
        errors="coerce",
    )

    data["sequence"] = pd.to_numeric(
        data["sequence"],
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
        .apply(normalize_condition)
        .astype("string")
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    before_drop = len(data)

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    print(
        "Lignes retirées pour donnée essentielle manquante :",
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

    if REFERENCE_CONDITION not in observed_conditions:
        raise ValueError(
            "La condition de référence est absente : "
            f"{REFERENCE_CONDITION}"
        )

    # ------------------------------------------------------------------
    # Contrôle du nombre de participants et d'items
    # ------------------------------------------------------------------

    number_of_subjects = (
        data["subject_id"].nunique()
    )

    number_of_items = (
        data["item_id"].nunique()
    )

    if number_of_subjects < 2:
        raise ValueError(
            "Au moins deux participants sont nécessaires."
        )

    if number_of_items < 2:
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
    ) / SEQUENCE_SCALE

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
        number_of_subjects,
    )

    print(
        "Nombre d'items :",
        number_of_items,
    )

    print(
        "Moyenne de la séquence utilisée pour le centrage :",
        sequence_mean,
    )

    print(
        "Une unité de sequence_c10 représente",
        SEQUENCE_SCALE,
        "essais.",
    )

    condition_counts = (
        data[[
            "subject_id",
            "condition",
        ]]
        .drop_duplicates()
        ["condition"]
        .value_counts()
    )

    print("")
    print(
        "Participants par condition :"
    )

    for condition, count in condition_counts.items():
        print(
            f"  {condition} :",
            int(count),
        )

    return (
        data,
        sequence_mean,
    )


# ============================================================================
# CONSTRUCTION DU MODÈLE
# ============================================================================

def build_model(
    data,
    formula,
):
    """
    Construit un modèle mixte avec des intercepts aléatoires croisés
    pour les participants et les items.
    """
    model_data = data.copy()

    # Statsmodels exige une variable groups.
    # Toutes les observations sont placées dans un groupe artificiel
    # unique. Les participants et les items sont ensuite définis comme
    # composantes de variance.
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
    model_name,
    reml,
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
# EFFETS FIXES
# ============================================================================

def create_fixed_effects_table(result):
    """
    Extrait les effets fixes du modèle REML.

    Pour chaque coefficient, un test de Wald bilatéral est calculé :

        z = estimation / erreur standard

        p = 2 × P(Z > |z|)
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
# COMPOSANTES DE VARIANCE
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

    component_map = {
        str(name).strip().lower():
            float(value)

        for name, value in zip(
            component_names,
            component_values,
        )
    }

    participant_variance = np.nan
    item_variance = np.nan

    for name, variance in component_map.items():
        if "subject" in name:
            participant_variance = variance

        elif "item" in name:
            item_variance = variance

    if (
        pd.isna(participant_variance)
        or pd.isna(item_variance)
    ):
        raise RuntimeError(
            "Impossible d'identifier les composantes "
            "participant et item. "
            f"Composantes trouvées : {component_map}"
        )

    return {
        "participant_variance":
            participant_variance,

        "item_variance":
            item_variance,

        "residual_variance":
            float(
                result.scale
            ),
    }


# ============================================================================
# R² ET MÉTRIQUES DU MODÈLE
# ============================================================================

def calculate_model_metrics(result):
    """
    Calcule les composantes de variance et les R² du modèle REML.

    R² marginal :
        proportion de variance associée aux effets fixes.

    R² conditionnel :
        proportion de variance associée aux effets fixes et aux
        effets aléatoires.
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

    random_total_variance = (
        participant_variance
        + item_variance
        + residual_variance
    )

    total_variance = (
        fixed_effect_variance
        + random_total_variance
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
                / random_total_variance
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
                / random_total_variance
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
                / random_total_variance
            ),

        "total_variance":
            total_variance,

        "marginal_r2":
            marginal_r2,

        "conditional_r2":
            conditional_r2,
    }


# ============================================================================
# COMPARAISON ML
# ============================================================================

def count_estimated_parameters(result):
    """
    Compte les paramètres figurant dans result.params.

    La différence entre les deux modèles est utilisée comme nombre
    de degrés de liberté du test LR.
    """
    return int(
        len(result.params)
    )


def compare_ml_models(
    null_result,
    null_optimizer,
    control_result,
    control_optimizer,
):
    """
    Compare le modèle nul et le modèle de contrôle en ML.

    LR = 2 × (logL_contrôle - logL_nul)
    """
    null_log_likelihood = safe_float(
        null_result.llf
    )

    control_log_likelihood = safe_float(
        control_result.llf
    )

    likelihood_ratio = (
        2
        * (
            control_log_likelihood
            - null_log_likelihood
        )
    )

    degrees_of_freedom = (
        count_estimated_parameters(
            control_result
        )
        - count_estimated_parameters(
            null_result
        )
    )

    if degrees_of_freedom <= 0:
        raise RuntimeError(
            "Le modèle de contrôle doit posséder plus de "
            "paramètres que le modèle nul."
        )

    p_value = stats.chi2.sf(
        likelihood_ratio,
        degrees_of_freedom,
    )

    return pd.DataFrame([
        {
            "model":
                "Null",

            "formula":
                NULL_FORMULA,

            "estimation":
                "ML",

            "converged":
                bool(
                    null_result.converged
                ),

            "optimizer":
                null_optimizer,

            "log_likelihood":
                null_log_likelihood,

            "aic":
                safe_float(
                    null_result.aic
                ),

            "bic":
                safe_float(
                    null_result.bic
                ),

            "number_of_estimated_parameters":
                count_estimated_parameters(
                    null_result
                ),

            "likelihood_ratio_vs_null":
                0.0,

            "degrees_of_freedom_difference":
                0,

            "likelihood_ratio_p_value":
                np.nan,
        },
        {
            "model":
                "Control",

            "formula":
                CONTROL_FORMULA,

            "estimation":
                "ML",

            "converged":
                bool(
                    control_result.converged
                ),

            "optimizer":
                control_optimizer,

            "log_likelihood":
                control_log_likelihood,

            "aic":
                safe_float(
                    control_result.aic
                ),

            "bic":
                safe_float(
                    control_result.bic
                ),

            "number_of_estimated_parameters":
                count_estimated_parameters(
                    control_result
                ),

            "likelihood_ratio_vs_null":
                likelihood_ratio,

            "degrees_of_freedom_difference":
                degrees_of_freedom,

            "likelihood_ratio_p_value":
                p_value,
        },
    ])


# ============================================================================
# SAUVEGARDE DU RÉSUMÉ REML
# ============================================================================

def save_reml_summary(
    result,
    optimizer,
    sequence_mean,
    metrics,
):
    """
    Sauvegarde le résumé du modèle de contrôle REML.
    """
    with open(
        REML_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(
            "MODÈLE MIXTE DE CONTRÔLE E1 — REML\n"
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
            CONTROL_FORMULA
        )

        output_file.write("\n\n")

        output_file.write(
            "Catégorie de référence de condition : "
            f"{REFERENCE_CONDITION}\n"
        )

        output_file.write(
            f"Centre de la séquence : {sequence_mean:.6f}\n"
        )

        output_file.write(
            f"Échelle de la séquence : {SEQUENCE_SCALE:.1f} essais\n"
        )

        output_file.write(
            f"Optimiseur retenu : {optimizer}\n"
        )

        output_file.write("\n")

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
    Supprime les anciennes sorties qui ne sont plus générées.
    """
    obsolete_filenames = [
        "null_model_ML_summary.txt",
        "control_model_ML_summary.txt",
        "control_model_REML_summary.txt",
        "model_comparison.csv",
        "control_model_fixed_effects.csv",
        "control_model_variance_components.csv",
        "variance_comparison.csv",
        "control_model_fit_statistics.csv",
        "control_model_predictions.csv",
        "condition_adjusted_means.csv",
        "sequence_predictions.csv",
        "control_model_residuals_vs_fitted.png",
        "control_model_residual_distribution.png",
        "control_model_qqplot.png",
        "control_model_condition_effect.png",
        "control_model_sequence_effect.png",
        "control_model_variance_comparison.png",
        "control_model_results.json",
        "control_model_report.txt",
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
        "MODÈLE MIXTE DE CONTRÔLE — EXPÉRIENCE E1"
    )

    remove_obsolete_output_files()

    try:
        # ==================================================================
        # 1. Chargement et préparation
        # ==================================================================

        (
            data,
            sequence_mean,
        ) = load_and_prepare_data()

        # ==================================================================
        # 2. Modèle nul ML
        # ==================================================================

        (
            null_ml,
            null_optimizer,
        ) = fit_model(
            data=data,
            formula=NULL_FORMULA,
            model_name="Modèle nul",
            reml=False,
        )

        # ==================================================================
        # 3. Modèle de contrôle ML
        # ==================================================================

        (
            control_ml,
            control_ml_optimizer,
        ) = fit_model(
            data=data,
            formula=CONTROL_FORMULA,
            model_name="Modèle de contrôle",
            reml=False,
        )

        # ==================================================================
        # 4. Modèle de contrôle REML
        # ==================================================================

        (
            control_reml,
            control_reml_optimizer,
        ) = fit_model(
            data=data,
            formula=CONTROL_FORMULA,
            model_name="Modèle de contrôle",
            reml=True,
        )

        # ==================================================================
        # 5. Coefficients REML
        # ==================================================================

        fixed_effects = (
            create_fixed_effects_table(
                control_reml
            )
        )

        fixed_effects.to_csv(
            FIXED_EFFECTS_FILE,
            index=False,
        )

        # ==================================================================
        # 6. Comparaison ML
        # ==================================================================

        model_comparison = (
            compare_ml_models(
                null_result=
                    null_ml,
                null_optimizer=
                    null_optimizer,
                control_result=
                    control_ml,
                control_optimizer=
                    control_ml_optimizer,
            )
        )

        model_comparison.to_csv(
            MODEL_COMPARISON_FILE,
            index=False,
        )

        # ==================================================================
        # 7. Variances et R²
        # ==================================================================

        metrics = calculate_model_metrics(
            control_reml
        )

        metrics_table = pd.DataFrame([
            {
                "model":
                    "Control_REML",

                **metrics,
            }
        ])

        metrics_table.to_csv(
            MODEL_METRICS_FILE,
            index=False,
        )

        # ==================================================================
        # 8. Résumé REML
        # ==================================================================

        save_reml_summary(
            result=control_reml,
            optimizer=
                control_reml_optimizer,
            sequence_mean=
                sequence_mean,
            metrics=metrics,
        )

        # ==================================================================
        # 9. Affichage des résultats
        # ==================================================================

        section(
            "COEFFICIENTS FIXES — REML"
        )

        print(
            fixed_effects
            .to_string(index=False)
        )

        section(
            "COMPARAISON ML"
        )

        print(
            model_comparison
            .to_string(index=False)
        )

        section(
            "VARIANCES ET R²"
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
            MODEL_COMPARISON_FILE,
            MODEL_METRICS_FILE,
        ]:
            print(
                output_file
            )

        print("")
        print("=" * 80)
        print("MODÈLE DE CONTRÔLE TERMINÉ")
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
