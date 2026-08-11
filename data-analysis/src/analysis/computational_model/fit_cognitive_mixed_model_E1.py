#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modèle mixte cognitif principal — Expérience E1

Variable dépendante :
    confidence

Effets fixes de contrôle :
    condition
    sequence_c10

Prédicteurs cognitifs :
    subject_accuracy_z
    item_entropy_z
    subject_mean_models_z
    models_within_subject_z
    validity_binary

Effets aléatoires croisés :
    intercept participant
    intercept item

Trois modèles sont ajustés en ML :
    1. Modèle nul
    2. Modèle de contrôle
    3. Modèle cognitif principal

Le modèle cognitif est également ajusté en REML pour la présentation
finale des coefficients et des composantes de variance.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import statsmodels
import statsmodels.formula.api as smf


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_analysis_E1.csv"
OUTPUT_DIR = BASE_DIR / "cognitive_mixed_model_E1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NULL_FORMULA = "confidence ~ 1"

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

# Effets aléatoires croisés participant et item.
VC_FORMULA = {
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}

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

OPTIMIZERS = ["lbfgs", "bfgs", "cg", "powell"]


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return np.nan


def standardize(series):
    """
    Standardisation avec écart-type empirique ddof=1.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=1)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(
            f"Impossible de standardiser {series.name}: écart-type={std}"
        )

    return (numeric - mean) / std, mean, std


def build_model(data, formula):
    """
    Construction d'un modèle à intercepts aléatoires croisés.

    Un groupe artificiel unique est utilisé, tandis que participant et item
    sont introduits comme composantes de variance.
    """
    model_data = data.copy()
    model_data["_global_group"] = 1

    return smf.mixedlm(
        formula=formula,
        data=model_data,
        groups=model_data["_global_group"],
        re_formula="0",
        vc_formula=VC_FORMULA,
    )


def fit_model(data, formula, reml, model_name):
    """
    Essaie plusieurs optimiseurs jusqu'à obtention d'un ajustement convergé.
    """
    section(
        f"AJUSTEMENT — {model_name} — "
        f"{'REML' if reml else 'ML'}"
    )

    print("Formule :", formula)

    last_result = None
    errors = []

    for optimizer in OPTIMIZERS:
        print(f"\nTentative avec l'optimiseur : {optimizer}")

        try:
            model = build_model(data, formula)

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                result = model.fit(
                    reml=reml,
                    method=optimizer,
                    maxiter=3000,
                    full_output=True,
                    disp=False,
                )

            last_result = result

            print("Convergence :", bool(result.converged))
            print("Log-vraisemblance :", result.llf)

            if caught_warnings:
                print("Avertissements :")
                for warning in caught_warnings:
                    print("  -", str(warning.message))

            if result.converged:
                print("Ajustement convergé avec :", optimizer)
                return result, optimizer, caught_warnings

        except Exception as exc:
            message = f"{optimizer}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print("Échec :", message)

    if last_result is not None:
        print(
            "\nATTENTION : aucun optimiseur n'a signalé une convergence "
            "complète. Le dernier résultat est retourné."
        )
        return last_result, "non_converged", []

    raise RuntimeError(
        "Échec de tous les optimiseurs :\n" + "\n".join(errors)
    )


def get_variance_components(result):
    """
    Extraction robuste des composantes de variance.
    """
    names = list(result.model.exog_vc.names)
    values = np.asarray(result.vcomp, dtype=float)

    variance_dict = dict(zip(names, values))

    return {
        "Participant": safe_float(variance_dict.get("subject")),
        "Item": safe_float(variance_dict.get("item")),
        "Residual": safe_float(result.scale),
    }


def fixed_effects_table(result):
    """
    Tableau des coefficients fixes.
    """
    names = list(result.fe_params.index)
    estimates = np.asarray(result.fe_params, dtype=float)

    covariance = result.cov_params().loc[names, names]
    standard_errors = np.sqrt(np.diag(covariance))

    z_values = estimates / standard_errors
    p_values = 2 * stats.norm.sf(np.abs(z_values))

    ci_lower = estimates - 1.96 * standard_errors
    ci_upper = estimates + 1.96 * standard_errors

    return pd.DataFrame({
        "parameter": names,
        "estimate": estimates,
        "standard_error": standard_errors,
        "z_value": z_values,
        "p_value": p_values,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    })


def model_statistics(result, model_name, formula):
    """
    Statistiques générales d'ajustement.
    """
    return {
        "model": model_name,
        "formula": formula,
        "converged": bool(result.converged),
        "log_likelihood": safe_float(result.llf),
        "aic": safe_float(result.aic),
        "bic": safe_float(result.bic),
        "number_of_estimated_parameters": len(result.params),
        "n_observations": int(result.nobs),
        "residual_variance": safe_float(result.scale),
    }


def likelihood_ratio_test(smaller_result, larger_result):
    """
    Test du rapport de vraisemblance entre deux modèles emboîtés ajustés en ML.
    """
    lr_statistic = 2 * (
        safe_float(larger_result.llf) - safe_float(smaller_result.llf)
    )

    df_difference = len(larger_result.params) - len(smaller_result.params)

    if df_difference > 0:
        p_value = stats.chi2.sf(lr_statistic, df_difference)
    else:
        p_value = np.nan

    return {
        "likelihood_ratio": lr_statistic,
        "degrees_of_freedom": df_difference,
        "p_value": p_value,
    }


def fixed_effect_variance(result):
    """
    Variance des prédictions dues uniquement aux effets fixes.
    """
    fixed_predictions = np.asarray(
        result.model.exog @ np.asarray(result.fe_params),
        dtype=float,
    )

    return float(np.var(fixed_predictions, ddof=1))


def calculate_r2(result):
    """
    R² marginal et conditionnel de Nakagawa, calculés à partir des
    composantes de variance du modèle linéaire mixte.
    """
    variances = get_variance_components(result)

    fixed_variance = fixed_effect_variance(result)
    participant_variance = variances["Participant"]
    item_variance = variances["Item"]
    residual_variance = variances["Residual"]

    total_variance = (
        fixed_variance
        + participant_variance
        + item_variance
        + residual_variance
    )

    marginal_r2 = fixed_variance / total_variance

    conditional_r2 = (
        fixed_variance
        + participant_variance
        + item_variance
    ) / total_variance

    return {
        "fixed_effect_variance": fixed_variance,
        "participant_variance": participant_variance,
        "item_variance": item_variance,
        "residual_variance": residual_variance,
        "total_variance": total_variance,
        "marginal_r2": marginal_r2,
        "conditional_r2": conditional_r2,
    }


def variance_comparison(results):
    """
    Comparaison des variances entre plusieurs modèles.
    """
    rows = []

    for model_name, result in results.items():
        variances = get_variance_components(result)

        structured_variance = (
            variances["Participant"] + variances["Item"]
        )
        total_random_variance = (
            structured_variance + variances["Residual"]
        )

        for component, variance in variances.items():
            rows.append({
                "model": model_name,
                "component": component,
                "variance": variance,
                "standard_deviation": np.sqrt(variance),
                "proportion_random_total": (
                    variance / total_random_variance
                ),
            })

    return pd.DataFrame(rows)


def coefficient_interpretation_table(fixed_table):
    """
    Ajoute des libellés interprétatifs simples aux coefficients.
    """
    interpretations = {
        "Intercept": (
            "Confiance prédite pour la condition Neutral, à la séquence "
            "moyenne, avec tous les prédicteurs standardisés à leur moyenne "
            "et validity_binary=0."
        ),
        "C(condition, Treatment(reference='Neutral'))[T.Standard]": (
            "Différence moyenne Standard moins Neutral."
        ),
        "sequence_c10": (
            "Variation de confiance pour dix essais supplémentaires."
        ),
        "subject_accuracy_z": (
            "Variation de confiance pour une augmentation d'un écart-type "
            "de la précision moyenne du participant."
        ),
        "item_entropy_z": (
            "Variation de confiance pour une augmentation d'un écart-type "
            "de l'entropie de l'item."
        ),
        "subject_mean_models_z": (
            "Effet interindividuel : variation de confiance pour une "
            "augmentation d'un écart-type du nombre moyen de modèles "
            "mentaux du participant."
        ),
        "models_within_subject_z": (
            "Effet intra-individuel : variation de confiance lorsque le "
            "nombre de modèles pour le type de tâche dépasse la moyenne "
            "personnelle d'un écart-type global."
        ),
        "validity_binary": (
            "Différence moyenne entre les essais valides et invalides, "
            "toutes les autres variables étant maintenues constantes."
        ),
    }

    output = fixed_table.copy()
    output["interpretation"] = output["parameter"].map(interpretations)

    return output


def create_predictions(data, result):
    """
    Prédictions marginales dues uniquement aux effets fixes.
    """
    predictions = data[[
        "subject_id",
        "item_id",
        "sequence",
        "condition",
        "confidence",
        "subject_accuracy_z",
        "item_entropy_z",
        "subject_mean_models_z",
        "models_within_subject_z",
        "validity_binary",
    ]].copy()

    fixed_pred = np.asarray(
        result.model.exog @ np.asarray(result.fe_params),
        dtype=float,
    )

    predictions["predicted_fixed"] = fixed_pred
    predictions["fixed_residual"] = (
        predictions["confidence"] - predictions["predicted_fixed"]
    )

    return predictions


# ============================================================================
# CHARGEMENT ET PRÉPARATION
# ============================================================================

section("MODÈLE MIXTE COGNITIF PRINCIPAL E1")

print("Version Python :", sys.version.split()[0])
print("Version pandas :", pd.__version__)
print("Version NumPy :", np.__version__)
print("Version SciPy :", scipy.__version__)
print("Version statsmodels :", statsmodels.__version__)

section("CHARGEMENT DES DONNÉES")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable : {DATA_FILE}")

data = pd.read_csv(DATA_FILE)

print("Fichier :", DATA_FILE)
print("Nombre de lignes brutes :", len(data))
print("Nombre de colonnes :", len(data.columns))

missing_columns = [
    column for column in ESSENTIAL_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes nécessaires absentes : "
        + ", ".join(missing_columns)
    )

if "analysis_complete" in data.columns:
    before = len(data)

    complete_mask = (
        data["analysis_complete"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    data = data.loc[complete_mask].copy()

    print(
        "Lignes retirées car analysis_complete=False :",
        before - len(data),
    )

for column in [
    "confidence",
    "sequence",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
    "validity_binary",
]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

before = len(data)
data = data.dropna(subset=ESSENTIAL_COLUMNS).copy()

print(
    "Lignes supprimées pour donnée essentielle manquante :",
    before - len(data),
)

data["subject_id"] = data["subject_id"].astype(str)
data["item_id"] = data["item_id"].astype(str)
data["condition"] = data["condition"].astype(str).str.strip()

valid_conditions = {"Neutral", "Standard"}
observed_conditions = set(data["condition"].unique())

if not observed_conditions.issubset(valid_conditions):
    raise ValueError(
        f"Conditions inattendues : {sorted(observed_conditions)}"
    )

data["validity_binary"] = data["validity_binary"].astype(int)

if not set(data["validity_binary"].unique()).issubset({0, 1}):
    raise ValueError("validity_binary doit uniquement contenir 0 et 1.")

sequence_mean = data["sequence"].mean()
data["sequence_c10"] = (
    data["sequence"] - sequence_mean
) / 10.0

standardization_rows = []

for column in STANDARDIZED_COLUMNS:
    z_column = f"{column}_z"

    data[z_column], mean, std = standardize(data[column])

    standardization_rows.append({
        "variable": column,
        "standardized_variable": z_column,
        "mean": mean,
        "standard_deviation": std,
    })

standardization_table = pd.DataFrame(standardization_rows)

print("Nombre de lignes utilisées :", len(data))
print("Nombre de participants :", data["subject_id"].nunique())
print("Nombre d'items :", data["item_id"].nunique())
print("Moyenne de confidence :", data["confidence"].mean())
print("Moyenne de sequence :", sequence_mean)

print("\nStatistiques de standardisation :")
print(standardization_table.to_string(index=False))

standardization_table.to_csv(
    OUTPUT_DIR / "predictor_standardization.csv",
    index=False,
)


# ============================================================================
# VÉRIFICATIONS DES PRÉDICTEURS
# ============================================================================

section("VÉRIFICATIONS DES PRÉDICTEURS")

predictor_columns = [
    "sequence_c10",
    "subject_accuracy_z",
    "item_entropy_z",
    "subject_mean_models_z",
    "models_within_subject_z",
    "validity_binary",
]

correlation_table = data[predictor_columns].corr()

print(correlation_table.round(4).to_string())

correlation_table.to_csv(
    OUTPUT_DIR / "cognitive_predictor_correlations.csv"
)

high_correlations = []

for i, first in enumerate(predictor_columns):
    for second in predictor_columns[i + 1:]:
        correlation = correlation_table.loc[first, second]

        if abs(correlation) >= 0.80:
            high_correlations.append({
                "predictor_1": first,
                "predictor_2": second,
                "correlation": correlation,
            })

high_correlation_table = pd.DataFrame(high_correlations)

high_correlation_table.to_csv(
    OUTPUT_DIR / "high_predictor_correlations.csv",
    index=False,
)

if len(high_correlation_table) == 0:
    print("\nAucune corrélation absolue >= 0.80.")
else:
    print("\nATTENTION : corrélations élevées :")
    print(high_correlation_table.to_string(index=False))


# ============================================================================
# AJUSTEMENT DES MODÈLES EN ML
# ============================================================================

null_ml, null_optimizer, null_warnings = fit_model(
    data=data,
    formula=NULL_FORMULA,
    reml=False,
    model_name="Modèle nul",
)

control_ml, control_optimizer, control_warnings = fit_model(
    data=data,
    formula=CONTROL_FORMULA,
    reml=False,
    model_name="Modèle de contrôle",
)

cognitive_ml, cognitive_ml_optimizer, cognitive_ml_warnings = fit_model(
    data=data,
    formula=COGNITIVE_FORMULA,
    reml=False,
    model_name="Modèle cognitif principal",
)


# ============================================================================
# AJUSTEMENT FINAL EN REML
# ============================================================================

cognitive_reml, cognitive_reml_optimizer, cognitive_reml_warnings = fit_model(
    data=data,
    formula=COGNITIVE_FORMULA,
    reml=True,
    model_name="Modèle cognitif principal",
)


# ============================================================================
# SAUVEGARDE DES RÉSUMÉS
# ============================================================================

section("RÉSUMÉ DU MODÈLE COGNITIF — REML")

print(cognitive_reml.summary())

with open(
    OUTPUT_DIR / "cognitive_model_ML_summary.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(str(cognitive_ml.summary()))

with open(
    OUTPUT_DIR / "cognitive_model_REML_summary.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(str(cognitive_reml.summary()))


# ============================================================================
# COEFFICIENTS FIXES
# ============================================================================

section("EFFETS FIXES DU MODÈLE COGNITIF")

fixed_ml = fixed_effects_table(cognitive_ml)
fixed_reml = fixed_effects_table(cognitive_reml)

fixed_ml = coefficient_interpretation_table(fixed_ml)
fixed_reml = coefficient_interpretation_table(fixed_reml)

print(fixed_reml.to_string(index=False))

fixed_ml.to_csv(
    OUTPUT_DIR / "cognitive_model_fixed_effects_ML.csv",
    index=False,
)

fixed_reml.to_csv(
    OUTPUT_DIR / "cognitive_model_fixed_effects_REML.csv",
    index=False,
)


# ============================================================================
# COMPARAISON DES MODÈLES
# ============================================================================

section("COMPARAISON DES MODÈLES EN ML")

models = [
    ("Null", NULL_FORMULA, null_ml),
    ("Control", CONTROL_FORMULA, control_ml),
    ("Cognitive", COGNITIVE_FORMULA, cognitive_ml),
]

comparison_rows = [
    model_statistics(result, name, formula)
    for name, formula, result in models
]

comparison_table = pd.DataFrame(comparison_rows)

null_vs_control = likelihood_ratio_test(null_ml, control_ml)
control_vs_cognitive = likelihood_ratio_test(control_ml, cognitive_ml)
null_vs_cognitive = likelihood_ratio_test(null_ml, cognitive_ml)

lr_table = pd.DataFrame([
    {
        "comparison": "Null vs Control",
        **null_vs_control,
    },
    {
        "comparison": "Control vs Cognitive",
        **control_vs_cognitive,
    },
    {
        "comparison": "Null vs Cognitive",
        **null_vs_cognitive,
    },
])

print("\nStatistiques d'ajustement :")
print(comparison_table.to_string(index=False))

print("\nTests du rapport de vraisemblance :")
print(lr_table.to_string(index=False))

comparison_table.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)

lr_table.to_csv(
    OUTPUT_DIR / "likelihood_ratio_tests.csv",
    index=False,
)


# ============================================================================
# COMPOSANTES DE VARIANCE
# ============================================================================

section("COMPARAISON DES COMPOSANTES DE VARIANCE")

variance_table = variance_comparison({
    "Null_ML": null_ml,
    "Control_ML": control_ml,
    "Cognitive_ML": cognitive_ml,
    "Cognitive_REML": cognitive_reml,
})

print(variance_table.to_string(index=False))

variance_table.to_csv(
    OUTPUT_DIR / "variance_components.csv",
    index=False,
)


# ============================================================================
# R²
# ============================================================================

section("R² DES MODÈLES")

r2_rows = []

for model_name, result in [
    ("Null_ML", null_ml),
    ("Control_ML", control_ml),
    ("Cognitive_ML", cognitive_ml),
    ("Cognitive_REML", cognitive_reml),
]:
    values = calculate_r2(result)
    values["model"] = model_name
    r2_rows.append(values)

r2_table = pd.DataFrame(r2_rows)

r2_table = r2_table[[
    "model",
    "fixed_effect_variance",
    "participant_variance",
    "item_variance",
    "residual_variance",
    "total_variance",
    "marginal_r2",
    "conditional_r2",
]]

print(r2_table.to_string(index=False))

r2_table.to_csv(
    OUTPUT_DIR / "model_r2.csv",
    index=False,
)


# ============================================================================
# PRÉDICTIONS
# ============================================================================

section("PRÉDICTIONS DU MODÈLE COGNITIF")

predictions = create_predictions(data, cognitive_reml)

rmse_fixed = np.sqrt(
    np.mean(predictions["fixed_residual"] ** 2)
)

mae_fixed = np.mean(
    np.abs(predictions["fixed_residual"])
)

print("RMSE des prédictions fixes :", rmse_fixed)
print("MAE des prédictions fixes :", mae_fixed)

predictions.to_csv(
    OUTPUT_DIR / "cognitive_model_predictions.csv",
    index=False,
)


# ============================================================================
# RÉSULTATS JSON
# ============================================================================

results_json = {
    "data_file": str(DATA_FILE),
    "output_directory": str(OUTPUT_DIR),
    "n_observations": int(len(data)),
    "n_subjects": int(data["subject_id"].nunique()),
    "n_items": int(data["item_id"].nunique()),
    "sequence_center": safe_float(sequence_mean),
    "formulas": {
        "null": NULL_FORMULA,
        "control": CONTROL_FORMULA,
        "cognitive": COGNITIVE_FORMULA,
    },
    "optimizers": {
        "null_ml": null_optimizer,
        "control_ml": control_optimizer,
        "cognitive_ml": cognitive_ml_optimizer,
        "cognitive_reml": cognitive_reml_optimizer,
    },
    "convergence": {
        "null_ml": bool(null_ml.converged),
        "control_ml": bool(control_ml.converged),
        "cognitive_ml": bool(cognitive_ml.converged),
        "cognitive_reml": bool(cognitive_reml.converged),
    },
    "likelihood_ratio_tests": {
        "null_vs_control": {
            key: safe_float(value)
            for key, value in null_vs_control.items()
        },
        "control_vs_cognitive": {
            key: safe_float(value)
            for key, value in control_vs_cognitive.items()
        },
        "null_vs_cognitive": {
            key: safe_float(value)
            for key, value in null_vs_cognitive.items()
        },
    },
    "prediction_metrics_fixed_only": {
        "rmse": safe_float(rmse_fixed),
        "mae": safe_float(mae_fixed),
    },
}

with open(
    OUTPUT_DIR / "cognitive_model_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(results_json, file, indent=4, ensure_ascii=False)


# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

section("RÉSUMÉ FINAL")

print("Modèle cognitif principal ajusté avec succès.")
print("Formule finale :")
print(COGNITIVE_FORMULA)

print("\nFichiers principaux à examiner :")
print(
    OUTPUT_DIR / "cognitive_model_fixed_effects_REML.csv"
)
print(
    OUTPUT_DIR / "model_comparison.csv"
)
print(
    OUTPUT_DIR / "likelihood_ratio_tests.csv"
)
print(
    OUTPUT_DIR / "variance_components.csv"
)
print(
    OUTPUT_DIR / "model_r2.csv"
)
print(
    OUTPUT_DIR / "cognitive_predictor_correlations.csv"
)
print(
    OUTPUT_DIR / "cognitive_model_REML_summary.txt"
)
