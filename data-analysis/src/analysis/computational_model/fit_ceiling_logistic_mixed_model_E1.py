#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GLMM logistique bayésien de l'utilisation du plafond de confiance — E1.

Variable dépendante :
    at_ceiling = 1 si confidence == 100
                 0 sinon

Effets fixes :
    condition
    sequence_c10
    subject_accuracy_z
    item_entropy_z
    subject_mean_models_z
    models_within_subject_z

Effets aléatoires croisés :
    intercept participant
    intercept item

Estimation :
    approximation variationnelle bayésienne avec
    statsmodels.BinomialBayesMixedGLM.fit_vb()

Les coefficients sont exprimés en log-odds.
Le script produit également les odds ratios et des intervalles
postérieurs approximatifs à 95 %.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit

import statsmodels
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_analysis_E1.csv"

OUTPUT_DIR = BASE_DIR / "ceiling_logistic_mixed_model_E1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORMULA = (
    "at_ceiling ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z"
)

# Chaque formule définit une famille d'effets aléatoires partageant
# le même paramètre de variance.
VC_FORMULAS = {
    "participant": "0 + C(subject_id)",
    "item": "0 + C(item_id)",
}

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

# Priors utilisés par BinomialBayesMixedGLM.
#
# fe_p : écart-type a priori des coefficients fixes.
# vcp_p : écart-type a priori des logarithmes des écarts-types aléatoires.
#
# vcp_p=0.5 est un choix modérément régularisant qui facilite généralement
# l'estimation des composantes de variance binomiales.
FE_P = 2.0
VCP_P = 0.5


# ============================================================================
# FONCTIONS
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
    Standardisation utilisant l'écart-type empirique ddof=1.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    mean = numeric.mean()
    standard_deviation = numeric.std(ddof=1)

    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError(
            f"Impossible de standardiser {series.name} : "
            f"écart-type={standard_deviation}"
        )

    standardized = (
        numeric - mean
    ) / standard_deviation

    return standardized, mean, standard_deviation


def posterior_fixed_effects_table(result):
    """
    Tableau des effets fixes.

    fe_mean :
        moyenne postérieure approximative du coefficient en log-odds.

    fe_sd :
        écart-type postérieur approximatif.

    Les intervalles sont calculés comme moyenne +/- 1.96 SD.
    """
    names = list(result.model.fep_names)

    means = np.asarray(result.fe_mean, dtype=float)
    standard_deviations = np.asarray(result.fe_sd, dtype=float)

    lower = means - 1.96 * standard_deviations
    upper = means + 1.96 * standard_deviations

    odds_ratios = np.exp(means)
    odds_ratio_lower = np.exp(lower)
    odds_ratio_upper = np.exp(upper)

    probability_positive = np.full(len(means), np.nan)
    probability_negative = np.full(len(means), np.nan)

    for index, (mean, sd) in enumerate(
        zip(means, standard_deviations)
    ):
        if sd > 0 and np.isfinite(sd):
            # Approximation normale de la distribution postérieure.
            z_zero = (0 - mean) / sd

            from scipy.stats import norm

            probability_negative[index] = norm.cdf(z_zero)
            probability_positive[index] = 1 - norm.cdf(z_zero)

    table = pd.DataFrame({
        "parameter": names,
        "posterior_mean_log_odds": means,
        "posterior_sd": standard_deviations,
        "credible_95_lower_log_odds": lower,
        "credible_95_upper_log_odds": upper,
        "odds_ratio": odds_ratios,
        "credible_95_lower_odds_ratio": odds_ratio_lower,
        "credible_95_upper_odds_ratio": odds_ratio_upper,
        "posterior_probability_positive": probability_positive,
        "posterior_probability_negative": probability_negative,
    })

    return table


def random_effect_sd_table(result):
    """
    Les paramètres vcp sont exprimés comme le logarithme de l'écart-type
    de chaque famille d'effets aléatoires.

    exp(vcp_mean) fournit donc l'écart-type aléatoire estimé sur
    l'échelle logit.
    """
    names = list(result.model.vcp_names)

    log_sd_mean = np.asarray(result.vcp_mean, dtype=float)
    log_sd_posterior_sd = np.asarray(result.vcp_sd, dtype=float)

    log_sd_lower = (
        log_sd_mean - 1.96 * log_sd_posterior_sd
    )
    log_sd_upper = (
        log_sd_mean + 1.96 * log_sd_posterior_sd
    )

    random_sd = np.exp(log_sd_mean)
    random_sd_lower = np.exp(log_sd_lower)
    random_sd_upper = np.exp(log_sd_upper)

    return pd.DataFrame({
        "component": names,
        "posterior_mean_log_standard_deviation": log_sd_mean,
        "posterior_sd_log_standard_deviation":
            log_sd_posterior_sd,
        "random_effect_standard_deviation_logit": random_sd,
        "credible_95_lower_standard_deviation":
            random_sd_lower,
        "credible_95_upper_standard_deviation":
            random_sd_upper,
        "approximate_variance_logit": random_sd ** 2,
    })


def fixed_effect_predictions(result):
    """
    Probabilités prédites à partir des effets fixes seulement.
    """
    linear_predictor = (
        result.model.exog
        @ np.asarray(result.fe_mean, dtype=float)
    )

    probabilities = expit(linear_predictor)

    return linear_predictor, probabilities


def calculate_brier_score(observed, probability):
    observed = np.asarray(observed, dtype=float)
    probability = np.asarray(probability, dtype=float)

    return float(
        np.mean((observed - probability) ** 2)
    )


def calculate_log_loss(observed, probability):
    observed = np.asarray(observed, dtype=float)
    probability = np.asarray(probability, dtype=float)

    epsilon = 1e-15

    probability = np.clip(
        probability,
        epsilon,
        1 - epsilon,
    )

    return float(
        -np.mean(
            observed * np.log(probability)
            + (1 - observed) * np.log(1 - probability)
        )
    )


def calibration_table(observed, probability, n_bins=10):
    """
    Tableau descriptif de calibration des prédictions fixes.
    """
    calibration_data = pd.DataFrame({
        "observed": np.asarray(observed, dtype=int),
        "predicted_probability": np.asarray(
            probability,
            dtype=float,
        ),
    })

    try:
        calibration_data["probability_bin"] = pd.qcut(
            calibration_data["predicted_probability"],
            q=n_bins,
            duplicates="drop",
        )
    except ValueError:
        calibration_data["probability_bin"] = pd.cut(
            calibration_data["predicted_probability"],
            bins=n_bins,
            duplicates="drop",
        )

    summary = (
        calibration_data
        .groupby(
            "probability_bin",
            observed=False,
        )
        .agg(
            n_observations=("observed", "size"),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_ceiling_rate=("observed", "mean"),
        )
        .reset_index()
    )

    summary["probability_bin"] = (
        summary["probability_bin"].astype(str)
    )

    return summary


def condition_summary(data):
    """
    Taux brut de réponses à 100 dans chaque condition.
    """
    return (
        data
        .groupby("condition", as_index=False)
        .agg(
            n_observations=("at_ceiling", "size"),
            n_at_ceiling=("at_ceiling", "sum"),
            ceiling_rate=("at_ceiling", "mean"),
            mean_confidence=("confidence", "mean"),
        )
    )


def sequence_summary(data):
    """
    Évolution descriptive du plafond par blocs de huit essais.
    """
    output = data.copy()

    output["sequence_block"] = pd.cut(
        output["sequence"],
        bins=[
            0,
            8,
            16,
            24,
            32,
            40,
            48,
            56,
            64,
        ],
        labels=[
            "01-08",
            "09-16",
            "17-24",
            "25-32",
            "33-40",
            "41-48",
            "49-56",
            "57-64",
        ],
        include_lowest=True,
    )

    return (
        output
        .groupby(
            ["condition", "sequence_block"],
            observed=False,
            as_index=False,
        )
        .agg(
            n_observations=("at_ceiling", "size"),
            n_at_ceiling=("at_ceiling", "sum"),
            ceiling_rate=("at_ceiling", "mean"),
        )
    )


def adjusted_scenario_predictions(result):
    """
    Probabilités ajustées pour quelques scénarios théoriques.

    Tous les prédicteurs cognitifs sont fixés à leur moyenne
    standardisée, donc à zéro.
    """
    fixed_table = posterior_fixed_effects_table(result)

    coefficients = dict(zip(
        fixed_table["parameter"],
        fixed_table["posterior_mean_log_odds"],
    ))

    intercept = coefficients.get("Intercept", 0.0)

    condition_name = (
        "C(condition, Treatment(reference='Neutral'))"
        "[T.Standard]"
    )

    condition_coefficient = coefficients.get(
        condition_name,
        0.0,
    )

    sequence_coefficient = coefficients.get(
        "sequence_c10",
        0.0,
    )

    # sequence_c10 vaut approximativement -3.15 au premier essai
    # et +3.15 au dernier essai lorsque la moyenne est 32.5.
    sequence_values = {
        "Beginning_sequence_1": (1 - 32.5) / 10,
        "Middle_sequence_32_5": 0.0,
        "End_sequence_64": (64 - 32.5) / 10,
    }

    rows = []

    for condition in ["Neutral", "Standard"]:
        condition_value = (
            condition_coefficient
            if condition == "Standard"
            else 0.0
        )

        for position_name, sequence_value in (
            sequence_values.items()
        ):
            linear_predictor = (
                intercept
                + condition_value
                + sequence_coefficient * sequence_value
            )

            rows.append({
                "condition": condition,
                "sequence_position": position_name,
                "sequence_c10": sequence_value,
                "predicted_log_odds_fixed_only":
                    linear_predictor,
                "predicted_probability_fixed_only":
                    expit(linear_predictor),
            })

    return pd.DataFrame(rows)


# ============================================================================
# INFORMATIONS SUR L'ENVIRONNEMENT
# ============================================================================

section("GLMM LOGISTIQUE DU PLAFOND DE CONFIANCE — E1")

print("Version Python :", sys.version.split()[0])
print("Version pandas :", pd.__version__)
print("Version NumPy :", np.__version__)
print("Version SciPy :", scipy.__version__)
print("Version statsmodels :", statsmodels.__version__)


# ============================================================================
# CHARGEMENT
# ============================================================================

section("CHARGEMENT DES DONNÉES")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {DATA_FILE}"
    )

data = pd.read_csv(DATA_FILE)

print("Fichier :", DATA_FILE)
print("Nombre de lignes brutes :", len(data))
print("Nombre de colonnes :", len(data.columns))

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
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

for column in NUMERIC_COLUMNS:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

before = len(data)

data = data.dropna(
    subset=REQUIRED_COLUMNS
).copy()

print(
    "Lignes supprimées pour donnée manquante :",
    before - len(data),
)

data["subject_id"] = (
    data["subject_id"].astype(str)
)

data["item_id"] = (
    data["item_id"].astype(str)
)

data["condition"] = (
    data["condition"]
    .astype(str)
    .str.strip()
)

observed_conditions = set(
    data["condition"].unique()
)

expected_conditions = {
    "Neutral",
    "Standard",
}

if observed_conditions != expected_conditions:
    raise ValueError(
        "Conditions observées inattendues : "
        + str(sorted(observed_conditions))
    )

if not data["confidence"].between(0, 100).all():
    raise ValueError(
        "Certaines valeurs de confidence sont hors de [0,100]."
    )

data["at_ceiling"] = (
    data["confidence"] == 100
).astype(int)

if data["at_ceiling"].nunique() != 2:
    raise ValueError(
        "La variable at_ceiling ne contient pas les deux modalités."
    )


# ============================================================================
# STANDARDISATION
# ============================================================================

section("PRÉPARATION DES PRÉDICTEURS")

sequence_mean = data["sequence"].mean()

data["sequence_c10"] = (
    data["sequence"] - sequence_mean
) / 10.0

standardization_rows = []

for variable in STANDARDIZED_VARIABLES:
    standardized_name = variable + "_z"

    (
        data[standardized_name],
        variable_mean,
        variable_standard_deviation,
    ) = standardize(data[variable])

    standardization_rows.append({
        "variable": variable,
        "standardized_variable": standardized_name,
        "mean": variable_mean,
        "standard_deviation":
            variable_standard_deviation,
    })

standardization = pd.DataFrame(
    standardization_rows
)

standardization.to_csv(
    OUTPUT_DIR / "predictor_standardization.csv",
    index=False,
)

print("Nombre de lignes utilisées :", len(data))
print(
    "Nombre de participants :",
    data["subject_id"].nunique(),
)
print(
    "Nombre d'items :",
    data["item_id"].nunique(),
)
print("Centre de sequence :", sequence_mean)
print("Taux global au plafond :", data["at_ceiling"].mean())


# ============================================================================
# DESCRIPTIFS
# ============================================================================

section("STATISTIQUES DESCRIPTIVES DU PLAFOND")

global_summary = pd.DataFrame([{
    "n_total": len(data),
    "n_at_ceiling": int(data["at_ceiling"].sum()),
    "ceiling_rate": data["at_ceiling"].mean(),
    "n_below_ceiling": int(
        (data["at_ceiling"] == 0).sum()
    ),
}])

condition_descriptive = condition_summary(data)
sequence_descriptive = sequence_summary(data)

print("\nRésumé global :")
print(global_summary.to_string(index=False))

print("\nRésumé par condition :")
print(condition_descriptive.to_string(index=False))

print("\nRésumé par blocs de séquence :")
print(sequence_descriptive.to_string(index=False))

global_summary.to_csv(
    OUTPUT_DIR / "ceiling_global_summary.csv",
    index=False,
)

condition_descriptive.to_csv(
    OUTPUT_DIR / "ceiling_by_condition.csv",
    index=False,
)

sequence_descriptive.to_csv(
    OUTPUT_DIR / "ceiling_by_sequence_block.csv",
    index=False,
)


# ============================================================================
# CONSTRUCTION DU MODÈLE
# ============================================================================

section("CONSTRUCTION DU MODÈLE")

print("Formule fixe :")
print(FORMULA)

print("\nComposantes aléatoires :")
for name, formula in VC_FORMULAS.items():
    print(f"  {name}: {formula}")

print("\nPrior fe_p :", FE_P)
print("Prior vcp_p :", VCP_P)

model = BinomialBayesMixedGLM.from_formula(
    formula=FORMULA,
    vc_formulas=VC_FORMULAS,
    data=data,
    fe_p=FE_P,
    vcp_p=VCP_P,
)


# ============================================================================
# AJUSTEMENT VARIATIONNEL BAYÉSIEN — PLUSIEURS INITIALISATIONS
# ============================================================================

section("AJUSTEMENT VARIATIONNEL BAYÉSIEN")

attempts = [
    {
        "name": "BFGS_scale_fe_true_gtol_1e-5",
        "fit_method": "BFGS",
        "scale_fe": True,
        "minim_opts": {
            "maxiter": 10000,
            "gtol": 1e-5,
        },
    },
    {
        "name": "BFGS_scale_fe_true_gtol_1e-4",
        "fit_method": "BFGS",
        "scale_fe": True,
        "minim_opts": {
            "maxiter": 10000,
            "gtol": 1e-4,
        },
    },
    {
        "name": "L-BFGS-B_scale_fe_true",
        "fit_method": "L-BFGS-B",
        "scale_fe": True,
        "minim_opts": {
            "maxiter": 15000,
            "ftol": 1e-10,
            "gtol": 1e-5,
            "maxls": 50,
        },
    },
]

all_attempts = []
result = None
selected_attempt_name = None

for attempt_number, attempt in enumerate(attempts, start=1):

    print("\n" + "-" * 80)
    print(
        f"TENTATIVE {attempt_number}/{len(attempts)} : "
        f"{attempt['name']}"
    )
    print("-" * 80)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        candidate = model.fit_vb(
            fit_method=attempt["fit_method"],
            minim_opts=attempt["minim_opts"],
            scale_fe=attempt["scale_fe"],
            verbose=False,
        )

    retvals = getattr(candidate, "optim_retvals", {}) or {}

    success = bool(retvals.get("success", False))
    objective = retvals.get("fun", np.nan)
    iterations = retvals.get("nit", np.nan)
    message = str(retvals.get("message", ""))

    gradient = retvals.get("jac", None)

    if gradient is not None:
        gradient = np.asarray(gradient, dtype=float)
        max_abs_gradient = float(
            np.max(np.abs(gradient))
        )
    else:
        max_abs_gradient = np.nan

    warning_messages = [
        str(warning.message)
        for warning in caught
    ]

    all_attempts.append({
        "attempt": attempt["name"],
        "success": success,
        "objective": objective,
        "iterations": iterations,
        "max_absolute_gradient": max_abs_gradient,
        "message": message,
        "warnings": " | ".join(warning_messages),
    })

    print("Succès :", success)
    print("Objectif :", objective)
    print("Itérations :", iterations)
    print("Gradient absolu maximal :", max_abs_gradient)
    print("Message :", message)

    if warning_messages:
        print("Avertissements :")
        for warning_message in warning_messages:
            print("  -", warning_message)

    # On conserve toujours la meilleure solution disponible.
    if result is None:
        result = candidate
        selected_attempt_name = attempt["name"]

    else:
        current_retvals = (
            getattr(result, "optim_retvals", {}) or {}
        )

        current_objective = current_retvals.get(
            "fun",
            np.inf,
        )

        if (
            np.isfinite(objective)
            and objective < current_objective
        ):
            result = candidate
            selected_attempt_name = attempt["name"]

    # Une convergence officielle est prioritaire.
    if success:
        result = candidate
        selected_attempt_name = attempt["name"]
        break


attempt_table = pd.DataFrame(all_attempts)

attempt_table.to_csv(
    OUTPUT_DIR / "ceiling_logistic_optimization_attempts.csv",
    index=False,
)

print("\n" + "=" * 80)
print("SOLUTION RETENUE")
print("=" * 80)
print("Tentative retenue :", selected_attempt_name)

optimization_results = (
    getattr(result, "optim_retvals", {}) or {}
)

optimization_success = bool(
    optimization_results.get("success", False)
)

optimization_message = str(
    optimization_results.get("message", "")
)

print("Succès :", optimization_success)
print("Message :", optimization_message)
print(result.summary())

# ============================================================================
# EXPORT DES EFFETS FIXES DU MODÈLE RETENU
# ============================================================================

fixed_effects = posterior_fixed_effects_table(result)

fixed_effects_file = (
    OUTPUT_DIR / "ceiling_logistic_fixed_effects.csv"
)

fixed_effects.to_csv(
    fixed_effects_file,
    index=False,
)

print("\nEffets fixes du modèle convergé :")
print(fixed_effects.to_string(index=False))

print("\nFichier créé :")
print(fixed_effects_file)


if not optimization_success:
    print(
        "\nATTENTION : aucune tentative n'a officiellement convergé."
    )
    print(
        "Les résultats doivent rester descriptifs tant qu'ils ne sont "
        "pas confirmés avec une autre méthode."
    )

with open(
    OUTPUT_DIR / "ceiling_logistic_model_summary.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(str(result.summary()))

    file.write("\n\nTentative retenue :\n")
    file.write(str(selected_attempt_name))

    file.write("\n\nOptimisation :\n")
    file.write(str(optimization_results))

    file.write("\n\nToutes les tentatives :\n")
    file.write(attempt_table.to_string(index=False))



# ============================================================================
# PRÉDICTIONS FIXES
# ============================================================================

section("PRÉDICTIONS FIXES")

(
    fixed_linear_predictor,
    fixed_probability,
) = fixed_effect_predictions(result)

prediction_table = data[[
    "subject_id",
    "item_id",
    "sequence",
    "condition",
    "confidence",
    "at_ceiling",
    "subject_accuracy_z",
    "item_entropy_z",
    "subject_mean_models_z",
    "models_within_subject_z",
]].copy()

prediction_table["predicted_log_odds_fixed_only"] = (
    fixed_linear_predictor
)

prediction_table["predicted_probability_fixed_only"] = (
    fixed_probability
)

prediction_table["prediction_error_fixed_only"] = (
    prediction_table["at_ceiling"]
    - prediction_table[
        "predicted_probability_fixed_only"
    ]
)

brier_score = calculate_brier_score(
    prediction_table["at_ceiling"],
    prediction_table[
        "predicted_probability_fixed_only"
    ],
)

log_loss = calculate_log_loss(
    prediction_table["at_ceiling"],
    prediction_table[
        "predicted_probability_fixed_only"
    ],
)

classification = (
    fixed_probability >= 0.5
).astype(int)

classification_accuracy = np.mean(
    classification
    == data["at_ceiling"].to_numpy()
)

print("Brier score, effets fixes seulement :", brier_score)
print("Log-loss, effets fixes seulement :", log_loss)
print(
    "Exactitude au seuil 0.5, effets fixes seulement :",
    classification_accuracy,
)

prediction_table.to_csv(
    OUTPUT_DIR / "ceiling_logistic_predictions.csv",
    index=False,
)


# ============================================================================
# CALIBRATION
# ============================================================================

section("CALIBRATION DES PRÉDICTIONS FIXES")

calibration = calibration_table(
    observed=data["at_ceiling"],
    probability=fixed_probability,
    n_bins=10,
)

print(calibration.to_string(index=False))

calibration.to_csv(
    OUTPUT_DIR / "ceiling_logistic_calibration.csv",
    index=False,
)


# ============================================================================
# SCÉNARIOS AJUSTÉS
# ============================================================================

section("PROBABILITÉS AJUSTÉES PAR CONDITION ET SÉQUENCE")

scenario_predictions = adjusted_scenario_predictions(
    result
)

print(scenario_predictions.to_string(index=False))

scenario_predictions.to_csv(
    OUTPUT_DIR / "adjusted_ceiling_probabilities.csv",
    index=False,
)


# ============================================================================
# SAUVEGARDE JSON
# ============================================================================

results_json = {
    "data_file": str(DATA_FILE),
    "output_directory": str(OUTPUT_DIR),
    "formula": FORMULA,
    "vc_formulas": VC_FORMULAS,
    "n_observations": int(len(data)),
    "n_subjects": int(data["subject_id"].nunique()),
    "n_items": int(data["item_id"].nunique()),
    "n_at_ceiling": int(data["at_ceiling"].sum()),
    "ceiling_rate": safe_float(
        data["at_ceiling"].mean()
    ),
    "sequence_center": safe_float(sequence_mean),
    "priors": {
        "fe_p": FE_P,
        "vcp_p": VCP_P,
    },
    "optimization": {
        "success": (
            bool(optimization_success)
            if optimization_success is not None
            else None
        ),
        "message": (
            str(optimization_message)
            if optimization_message is not None
            else None
        ),
        "iterations": (
            int(optimization_results["nit"])
            if "nit" in optimization_results
            else None
        ),
    },
    "prediction_metrics_fixed_only": {
        "brier_score": brier_score,
        "log_loss": log_loss,
        "classification_accuracy_threshold_0_5":
            safe_float(classification_accuracy),
    },
}

with open(
    OUTPUT_DIR / "ceiling_logistic_results.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results_json,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================================
# FIN
# ============================================================================

section("TERMINÉ")

print("Modèle ajusté.")
print("Résultats enregistrés dans :")
print(OUTPUT_DIR)

print("\nFichiers principaux :")
print(
    OUTPUT_DIR / "ceiling_logistic_fixed_effects.csv"
)
print(
    OUTPUT_DIR
    / "ceiling_logistic_random_effect_standard_deviations.csv"
)
print(
    OUTPUT_DIR / "adjusted_ceiling_probabilities.csv"
)
print(
    OUTPUT_DIR / "ceiling_by_condition.csv"
)
print(
    OUTPUT_DIR / "ceiling_by_sequence_block.csv"
)
print(
    OUTPUT_DIR / "ceiling_logistic_calibration.csv"
)
print(
    OUTPUT_DIR / "ceiling_logistic_model_summary.txt"
)
