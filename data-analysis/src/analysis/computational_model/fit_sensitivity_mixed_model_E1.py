#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses de sensibilité du modèle mixte E1.

Analyses réalisées :
1. Modèle de contrôle.
2. Modèle cognitif avec validity_binary.
3. Modèle cognitif alternatif avec task_type.
4. Tests drop-one de chaque prédicteur cognitif.

Les comparaisons de modèles sont réalisées en ML.
Les modèles finaux avec validité et task_type sont aussi ajustés en REML.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_analysis_E1.csv"
OUTPUT_DIR = BASE_DIR / "sensitivity_mixed_model_E1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VC_FORMULA = {
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}

CONTROL_TERMS = [
    "C(condition, Treatment(reference='Neutral'))",
    "sequence_c10",
]

COGNITIVE_TERMS = [
    "subject_accuracy_z",
    "item_entropy_z",
    "subject_mean_models_z",
    "models_within_subject_z",
]

OPTIMIZERS = ["lbfgs", "bfgs", "cg", "powell"]


# ============================================================================
# FONCTIONS
# ============================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def make_formula(terms):
    return "confidence ~ " + " + ".join(terms)


def standardize(series):
    series = pd.to_numeric(series, errors="coerce")
    mean = series.mean()
    std = series.std(ddof=1)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(
            f"Impossible de standardiser {series.name}: écart-type={std}"
        )

    return (series - mean) / std, mean, std


def build_model(data, formula):
    model_data = data.copy()
    model_data["_global_group"] = 1

    return smf.mixedlm(
        formula=formula,
        data=model_data,
        groups=model_data["_global_group"],
        re_formula="0",
        vc_formula=VC_FORMULA,
    )


def fit_model(data, formula, model_name, reml=False):
    section(
        f"AJUSTEMENT : {model_name} — "
        f"{'REML' if reml else 'ML'}"
    )

    print("Formule :", formula)

    errors = []

    for optimizer in OPTIMIZERS:
        print("Optimiseur :", optimizer)

        try:
            model = build_model(data, formula)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                result = model.fit(
                    reml=reml,
                    method=optimizer,
                    maxiter=3000,
                    full_output=True,
                    disp=False,
                )

            print("Convergence :", result.converged)
            print("Log-vraisemblance :", result.llf)

            for warning in caught:
                print("Avertissement :", warning.message)

            if result.converged:
                return result, optimizer

        except Exception as exc:
            message = f"{optimizer}: {type(exc).__name__}: {exc}"
            errors.append(message)
            print("Échec :", message)

    raise RuntimeError(
        f"Échec de l'ajustement de {model_name}:\n"
        + "\n".join(errors)
    )


def likelihood_ratio_test(reduced, full):
    lr = 2 * (full.llf - reduced.llf)
    df = len(full.params) - len(reduced.params)
    p_value = stats.chi2.sf(lr, df) if df > 0 else np.nan

    return lr, df, p_value


def fixed_effects_table(result):
    names = list(result.fe_params.index)
    estimates = np.asarray(result.fe_params, dtype=float)

    covariance = result.cov_params().loc[names, names]
    standard_errors = np.sqrt(np.diag(covariance))

    z_values = estimates / standard_errors
    p_values = 2 * stats.norm.sf(np.abs(z_values))

    return pd.DataFrame({
        "parameter": names,
        "estimate": estimates,
        "standard_error": standard_errors,
        "z_value": z_values,
        "p_value": p_values,
        "ci_95_lower": estimates - 1.96 * standard_errors,
        "ci_95_upper": estimates + 1.96 * standard_errors,
    })


def variance_components(result):
    names = list(result.model.exog_vc.names)
    values = np.asarray(result.vcomp, dtype=float)
    components = dict(zip(names, values))

    return {
        "participant_variance": components.get("subject", np.nan),
        "item_variance": components.get("item", np.nan),
        "residual_variance": result.scale,
    }


def fit_statistics(name, formula, result, method):
    variances = variance_components(result)

    return {
        "model": name,
        "method": method,
        "formula": formula,
        "converged": result.converged,
        "log_likelihood": result.llf,
        "aic": result.aic,
        "bic": result.bic,
        "n_parameters": len(result.params),
        "participant_variance": variances["participant_variance"],
        "item_variance": variances["item_variance"],
        "residual_variance": variances["residual_variance"],
    }


# ============================================================================
# CHARGEMENT
# ============================================================================

section("CHARGEMENT DES DONNÉES")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable : {DATA_FILE}")

data = pd.read_csv(DATA_FILE)

required_columns = [
    "confidence",
    "subject_id",
    "item_id",
    "condition",
    "sequence",
    "task_type",
    "validity_binary",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]

missing = [
    column for column in required_columns
    if column not in data.columns
]

if missing:
    raise ValueError(
        "Colonnes manquantes : " + ", ".join(missing)
    )

if "analysis_complete" in data.columns:
    complete = (
        data["analysis_complete"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    data = data.loc[complete].copy()

numeric_columns = [
    "confidence",
    "sequence",
    "validity_binary",
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

data = data.dropna(subset=required_columns).copy()

data["subject_id"] = data["subject_id"].astype(str)
data["item_id"] = data["item_id"].astype(str)

data["condition"] = (
    data["condition"].astype(str).str.strip()
)

data["task_type"] = (
    data["task_type"].astype(str).str.strip()
)

expected_tasks = {"MP", "MT", "AC", "DA"}
observed_tasks = set(data["task_type"].unique())

if observed_tasks != expected_tasks:
    raise ValueError(
        f"Types de tâches observés : {sorted(observed_tasks)}"
    )

sequence_mean = data["sequence"].mean()

data["sequence_c10"] = (
    data["sequence"] - sequence_mean
) / 10

standardization_rows = []

for variable in [
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]:
    standardized_name = variable + "_z"

    (
        data[standardized_name],
        variable_mean,
        variable_std,
    ) = standardize(data[variable])

    standardization_rows.append({
        "variable": variable,
        "mean": variable_mean,
        "standard_deviation": variable_std,
    })

pd.DataFrame(standardization_rows).to_csv(
    OUTPUT_DIR / "standardization.csv",
    index=False,
)

print("Lignes :", len(data))
print("Participants :", data["subject_id"].nunique())
print("Items :", data["item_id"].nunique())
print("Types de tâches :", sorted(observed_tasks))


# ============================================================================
# FORMULES
# ============================================================================

CONTROL_FORMULA = make_formula(CONTROL_TERMS)

BASE_COGNITIVE_FORMULA = make_formula(
    CONTROL_TERMS + COGNITIVE_TERMS
)

VALIDITY_FORMULA = make_formula(
    CONTROL_TERMS
    + COGNITIVE_TERMS
    + ["validity_binary"]
)

TASK_TYPE_FORMULA = make_formula(
    CONTROL_TERMS
    + COGNITIVE_TERMS
    + ["C(task_type, Treatment(reference='AC'))"]
)

section("FORMULES")

print("Contrôle :", CONTROL_FORMULA)
print("Cognitif sans validité :", BASE_COGNITIVE_FORMULA)
print("Cognitif avec validité :", VALIDITY_FORMULA)
print("Cognitif avec type de tâche :", TASK_TYPE_FORMULA)


# ============================================================================
# MODÈLES PRINCIPAUX ML
# ============================================================================

control_ml, optimizer_control = fit_model(
    data,
    CONTROL_FORMULA,
    "Contrôle",
    reml=False,
)

base_cognitive_ml, optimizer_base = fit_model(
    data,
    BASE_COGNITIVE_FORMULA,
    "Cognitif sans validité",
    reml=False,
)

validity_ml, optimizer_validity_ml = fit_model(
    data,
    VALIDITY_FORMULA,
    "Cognitif avec validité",
    reml=False,
)

task_type_ml, optimizer_task_ml = fit_model(
    data,
    TASK_TYPE_FORMULA,
    "Cognitif avec task_type",
    reml=False,
)


# ============================================================================
# MODÈLES FINAUX REML
# ============================================================================

validity_reml, optimizer_validity_reml = fit_model(
    data,
    VALIDITY_FORMULA,
    "Cognitif avec validité",
    reml=True,
)

task_type_reml, optimizer_task_reml = fit_model(
    data,
    TASK_TYPE_FORMULA,
    "Cognitif avec task_type",
    reml=True,
)


# ============================================================================
# TESTS GLOBAUX
# ============================================================================

section("TESTS GLOBAUX")

global_tests = []

comparisons = [
    (
        "Control vs Cognitive_without_validity",
        control_ml,
        base_cognitive_ml,
    ),
    (
        "Cognitive_without_validity vs Validity",
        base_cognitive_ml,
        validity_ml,
    ),
    (
        "Cognitive_without_validity vs Task_type",
        base_cognitive_ml,
        task_type_ml,
    ),
]

for comparison_name, reduced, full in comparisons:
    lr, df, p_value = likelihood_ratio_test(reduced, full)

    global_tests.append({
        "comparison": comparison_name,
        "likelihood_ratio": lr,
        "degrees_of_freedom": df,
        "p_value": p_value,
    })

global_tests = pd.DataFrame(global_tests)

print(global_tests.to_string(index=False))

global_tests.to_csv(
    OUTPUT_DIR / "global_likelihood_ratio_tests.csv",
    index=False,
)


# ============================================================================
# TESTS DROP-ONE
# ============================================================================

section("TESTS DROP-ONE DES PRÉDICTEURS COGNITIFS")

drop_one_rows = []

all_validity_terms = (
    CONTROL_TERMS
    + COGNITIVE_TERMS
    + ["validity_binary"]
)

tested_terms = (
    COGNITIVE_TERMS
    + ["validity_binary"]
)

for removed_term in tested_terms:
    reduced_terms = [
        term for term in all_validity_terms
        if term != removed_term
    ]

    reduced_formula = make_formula(reduced_terms)

    reduced_result, optimizer = fit_model(
        data,
        reduced_formula,
        f"Sans {removed_term}",
        reml=False,
    )

    lr, df, p_value = likelihood_ratio_test(
        reduced_result,
        validity_ml,
    )

    drop_one_rows.append({
        "removed_predictor": removed_term,
        "reduced_formula": reduced_formula,
        "likelihood_ratio": lr,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "reduced_aic": reduced_result.aic,
        "full_aic": validity_ml.aic,
        "delta_aic_reduced_minus_full": (
            reduced_result.aic - validity_ml.aic
        ),
        "optimizer": optimizer,
    })

drop_one_table = pd.DataFrame(drop_one_rows)

print(drop_one_table.to_string(index=False))

drop_one_table.to_csv(
    OUTPUT_DIR / "drop_one_tests.csv",
    index=False,
)


# ============================================================================
# COMPARAISON AIC/BIC
# ============================================================================

section("COMPARAISON AIC ET BIC")

fit_rows = [
    fit_statistics(
        "Control",
        CONTROL_FORMULA,
        control_ml,
        "ML",
    ),
    fit_statistics(
        "Cognitive_without_validity",
        BASE_COGNITIVE_FORMULA,
        base_cognitive_ml,
        "ML",
    ),
    fit_statistics(
        "Cognitive_validity",
        VALIDITY_FORMULA,
        validity_ml,
        "ML",
    ),
    fit_statistics(
        "Cognitive_task_type",
        TASK_TYPE_FORMULA,
        task_type_ml,
        "ML",
    ),
    fit_statistics(
        "Cognitive_validity",
        VALIDITY_FORMULA,
        validity_reml,
        "REML",
    ),
    fit_statistics(
        "Cognitive_task_type",
        TASK_TYPE_FORMULA,
        task_type_reml,
        "REML",
    ),
]

fit_table = pd.DataFrame(fit_rows)

print(fit_table.to_string(index=False))

fit_table.to_csv(
    OUTPUT_DIR / "model_fit_comparison.csv",
    index=False,
)


# ============================================================================
# COEFFICIENTS
# ============================================================================

fixed_effects_table(validity_reml).to_csv(
    OUTPUT_DIR / "validity_model_fixed_effects_REML.csv",
    index=False,
)

task_fixed = fixed_effects_table(task_type_reml)

print("\nEffets fixes du modèle task_type :")
print(task_fixed.to_string(index=False))

task_fixed.to_csv(
    OUTPUT_DIR / "task_type_model_fixed_effects_REML.csv",
    index=False,
)


# ============================================================================
# RÉSUMÉS COMPLETS
# ============================================================================

with open(
    OUTPUT_DIR / "validity_model_REML_summary.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(str(validity_reml.summary()))

with open(
    OUTPUT_DIR / "task_type_model_REML_summary.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(str(task_type_reml.summary()))


# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

section("TERMINÉ")

print("Résultats enregistrés dans :", OUTPUT_DIR)

print("\nFichiers à examiner :")
print(OUTPUT_DIR / "global_likelihood_ratio_tests.csv")
print(OUTPUT_DIR / "drop_one_tests.csv")
print(OUTPUT_DIR / "model_fit_comparison.csv")
print(OUTPUT_DIR / "task_type_model_fixed_effects_REML.csv")
print(OUTPUT_DIR / "task_type_model_REML_summary.txt")
