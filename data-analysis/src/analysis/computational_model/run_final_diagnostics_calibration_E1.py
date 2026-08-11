#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses finales E1 :

1. Ajustement du modèle linéaire mixte final parcimonieux.
2. Diagnostics des résidus.
3. Résumés des observations potentiellement influentes.
4. Stabilité leave-one-subject-out optionnelle.
5. Calibration métacognitive descriptive.
6. Modèle logistique mixte de l'exactitude en fonction de la confiance.
7. Figures et tableaux finaux.

ATTENTION :
    RUN_SUBJECT_JACKKNIFE = True entraîne 141 réajustements et peut être long.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)

import statsmodels
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_analysis_E1_n20.csv"

OUTPUT_DIR = BASE_DIR / "final_analysis_E1_n20"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
JACKKNIFE_DIR = OUTPUT_DIR / "jackknife"

for directory in [
    OUTPUT_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    DIAGNOSTIC_DIR,
    CALIBRATION_DIR,
    JACKKNIFE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# False pour un premier lancement rapide.
# True pour lancer 141 réajustements leave-one-subject-out.
RUN_SUBJECT_JACKKNIFE = True

# Si RUN_SUBJECT_JACKKNIFE=True, None signifie tous les participants.
# Pour tester le code rapidement, utiliser par exemple 10.
JACKKNIFE_MAX_SUBJECTS = None

# Nombre minimal d'essais corrects et incorrects pour calculer l'AUC
# individuelle.
MIN_CLASS_COUNT_FOR_AUC = 2

RANDOM_SEED = 20260730

FINAL_FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z"
)

CALIBRATION_FORMULA = (
    "is_correct_binary ~ "
    "confidence_z "
    "+ C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10"
)

VC_FORMULA_LINEAR = {
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}

VC_FORMULA_BINARY = {
    "participant": "0 + C(subject_id)",
    "item": "0 + C(item_id)",
}

REQUIRED_COLUMNS = [
    "subject_id",
    "item_id",
    "confidence",
    "is_correct",
    "condition",
    "sequence",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]

NUMERIC_COLUMNS = [
    "confidence",
    "is_correct",
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

LINEAR_OPTIMIZERS = [
    "lbfgs",
    "bfgs",
    "cg",
    "powell",
]

sns.set_theme(style="whitegrid", context="talk")
np.random.seed(RANDOM_SEED)


# =============================================================================
# OUTILS
# =============================================================================

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
    numeric = pd.to_numeric(series, errors="coerce")

    mean = numeric.mean()
    standard_deviation = numeric.std(ddof=1)

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        raise ValueError(
            f"Impossible de standardiser {series.name}: "
            f"écart-type={standard_deviation}"
        )

    standardized = (
        numeric - mean
    ) / standard_deviation

    return standardized, mean, standard_deviation


def save_figure(filename):
    path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Figure créée :", path)


def build_linear_model(data, formula):
    model_data = data.copy()
    model_data["_global_group"] = 1

    return smf.mixedlm(
        formula=formula,
        data=model_data,
        groups=model_data["_global_group"],
        re_formula="0",
        vc_formula=VC_FORMULA_LINEAR,
    )


def fit_linear_model(
    data,
    formula,
    reml=True,
    display=True,
):
    last_result = None
    errors = []

    for optimizer in LINEAR_OPTIMIZERS:
        if display:
            print("Tentative :", optimizer)

        try:
            model = build_linear_model(
                data=data,
                formula=formula,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                result = model.fit(
                    reml=reml,
                    method=optimizer,
                    maxiter=3000,
                    full_output=True,
                    disp=False,
                )

            last_result = result

            if display:
                print("Convergence :", result.converged)
                print("Log-vraisemblance :", result.llf)

                for warning in caught:
                    print(
                        "Avertissement :",
                        str(warning.message),
                    )

            if result.converged:
                return result, optimizer

        except Exception as exc:
            errors.append(
                f"{optimizer}: "
                f"{type(exc).__name__}: {exc}"
            )

            if display:
                print("Échec :", errors[-1])

    if last_result is not None:
        return last_result, "non_converged"

    raise RuntimeError(
        "Échec de tous les optimiseurs :\n"
        + "\n".join(errors)
    )


def linear_fixed_effects_table(result):
    names = list(result.fe_params.index)
    estimates = np.asarray(
        result.fe_params,
        dtype=float,
    )

    covariance = (
        result
        .cov_params()
        .loc[names, names]
    )

    standard_errors = np.sqrt(
        np.diag(covariance)
    )

    z_values = estimates / standard_errors
    p_values = 2 * stats.norm.sf(
        np.abs(z_values)
    )

    return pd.DataFrame({
        "parameter": names,
        "estimate": estimates,
        "standard_error": standard_errors,
        "z_value": z_values,
        "p_value": p_values,
        "ci_95_lower": (
            estimates - 1.96 * standard_errors
        ),
        "ci_95_upper": (
            estimates + 1.96 * standard_errors
        ),
    })


def get_variance_components(result):
    names = list(result.model.exog_vc.names)
    values = np.asarray(
        result.vcomp,
        dtype=float,
    )

    components = dict(zip(names, values))

    return {
        "participant_variance": safe_float(
            components.get("subject")
        ),
        "item_variance": safe_float(
            components.get("item")
        ),
        "residual_variance": safe_float(
            result.scale
        ),
    }


def get_conditional_predictions(result):
    """
    fittedvalues inclut normalement les effets fixes et les effets
    aléatoires estimés.

    En cas d'échec, revient aux prédictions des effets fixes seulement.
    """
    try:
        fitted = np.asarray(
            result.fittedvalues,
            dtype=float,
        )

        prediction_type = "conditional"
    except Exception:
        fitted = np.asarray(
            result.model.exog
            @ np.asarray(result.fe_params),
            dtype=float,
        )

        prediction_type = "fixed_only"

    return fitted, prediction_type


def normality_summary(values, name):
    values = pd.Series(values).dropna().to_numpy()

    skewness = stats.skew(
        values,
        bias=False,
    )

    excess_kurtosis = stats.kurtosis(
        values,
        fisher=True,
        bias=False,
    )

    # Shapiro n'est pas très informatif avec 9024 observations.
    # On le calcule sur un échantillon reproductible de 5000 au maximum.
    if len(values) > 5000:
        rng = np.random.default_rng(RANDOM_SEED)
        shapiro_values = rng.choice(
            values,
            size=5000,
            replace=False,
        )
    else:
        shapiro_values = values

    shapiro_statistic, shapiro_p = stats.shapiro(
        shapiro_values
    )

    return {
        "variable": name,
        "n": len(values),
        "mean": np.mean(values),
        "standard_deviation": np.std(
            values,
            ddof=1,
        ),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "shapiro_sample_n": len(shapiro_values),
        "shapiro_statistic": shapiro_statistic,
        "shapiro_p_value": shapiro_p,
    }


def posterior_binary_fixed_effects(result):
    names = list(result.model.fep_names)

    means = np.asarray(
        result.fe_mean,
        dtype=float,
    )

    posterior_sd = np.asarray(
        result.fe_sd,
        dtype=float,
    )

    lower = means - 1.96 * posterior_sd
    upper = means + 1.96 * posterior_sd

    return pd.DataFrame({
        "parameter": names,
        "posterior_mean_log_odds": means,
        "posterior_sd": posterior_sd,
        "credible_95_lower_log_odds": lower,
        "credible_95_upper_log_odds": upper,
        "odds_ratio": np.exp(means),
        "credible_95_lower_odds_ratio": np.exp(lower),
        "credible_95_upper_odds_ratio": np.exp(upper),
    })


# =============================================================================
# CHARGEMENT
# =============================================================================

section("CHARGEMENT DES DONNÉES")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Fichier absent : {DATA_FILE}"
    )

data = pd.read_csv(DATA_FILE)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes absentes : "
        + ", ".join(missing_columns)
    )

if "analysis_complete" in data.columns:
    complete_mask = (
        data["analysis_complete"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    data = data.loc[complete_mask].copy()

for column in NUMERIC_COLUMNS:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

data = data.dropna(
    subset=REQUIRED_COLUMNS
).copy()

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

data["is_correct_binary"] = (
    data["is_correct"].astype(int)
)

if not set(
    data["is_correct_binary"].unique()
).issubset({0, 1}):
    raise ValueError(
        "is_correct doit uniquement contenir 0 et 1."
    )

sequence_mean = data["sequence"].mean()

data["sequence_c10"] = (
    data["sequence"] - sequence_mean
) / 10

standardization_rows = []

for variable in STANDARDIZED_VARIABLES:
    z_name = variable + "_z"

    (
        data[z_name],
        variable_mean,
        variable_sd,
    ) = standardize(data[variable])

    standardization_rows.append({
        "variable": variable,
        "mean": variable_mean,
        "standard_deviation": variable_sd,
    })

data["confidence_probability"] = (
    data["confidence"] / 100
)

data["confidence_z"], confidence_mean, confidence_sd = (
    standardize(data["confidence"])
)

data["confidence_centered_50"] = (
    data["confidence"] - 50
) / 10

print("Lignes :", len(data))
print("Participants :", data["subject_id"].nunique())
print("Items :", data["item_id"].nunique())
print("Confiance moyenne :", confidence_mean)
print("Confiance SD :", confidence_sd)

pd.DataFrame(
    standardization_rows
).to_csv(
    TABLE_DIR / "standardization_parameters.csv",
    index=False,
)


# =============================================================================
# MODÈLE LINÉAIRE FINAL
# =============================================================================

section("MODÈLE LINÉAIRE FINAL")

final_result, final_optimizer = fit_linear_model(
    data=data,
    formula=FINAL_FORMULA,
    reml=True,
    display=True,
)

print(final_result.summary())

with open(
    OUTPUT_DIR / "final_linear_model_REML.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(str(final_result.summary()))

fixed_effects = linear_fixed_effects_table(
    final_result
)

fixed_effects.to_csv(
    TABLE_DIR / "final_linear_fixed_effects.csv",
    index=False,
)

variance_components = get_variance_components(
    final_result
)

variance_table = pd.DataFrame([
    {
        "component": "Participant",
        "variance": variance_components[
            "participant_variance"
        ],
    },
    {
        "component": "Item",
        "variance": variance_components[
            "item_variance"
        ],
    },
    {
        "component": "Residual",
        "variance": variance_components[
            "residual_variance"
        ],
    },
])

variance_table["standard_deviation"] = np.sqrt(
    variance_table["variance"]
)

variance_table["proportion"] = (
    variance_table["variance"]
    / variance_table["variance"].sum()
)

variance_table.to_csv(
    TABLE_DIR / "final_linear_variance_components.csv",
    index=False,
)


# =============================================================================
# DIAGNOSTICS DES RÉSIDUS
# =============================================================================

section("DIAGNOSTICS DES RÉSIDUS")

fitted, prediction_type = (
    get_conditional_predictions(final_result)
)

residuals = (
    data["confidence"].to_numpy()
    - fitted
)

residual_sd = np.std(
    residuals,
    ddof=1,
)

standardized_residuals = (
    residuals / residual_sd
)

diagnostic_data = data[[
    "subject_id",
    "item_id",
    "sequence",
    "condition",
    "confidence",
    "is_correct_binary",
]].copy()

diagnostic_data["fitted"] = fitted
diagnostic_data["residual"] = residuals
diagnostic_data["standardized_residual"] = (
    standardized_residuals
)
diagnostic_data["absolute_standardized_residual"] = (
    np.abs(standardized_residuals)
)

diagnostic_data.to_csv(
    DIAGNOSTIC_DIR / "observation_diagnostics.csv",
    index=False,
)

normality = pd.DataFrame([
    normality_summary(
        residuals,
        "conditional_residuals",
    )
])

normality.to_csv(
    DIAGNOSTIC_DIR / "residual_normality_summary.csv",
    index=False,
)

outlier_summary = pd.DataFrame([{
    "prediction_type": prediction_type,
    "n_observations": len(diagnostic_data),
    "n_abs_standardized_residual_gt_2": int(
        (
            diagnostic_data[
                "absolute_standardized_residual"
            ] > 2
        ).sum()
    ),
    "rate_abs_standardized_residual_gt_2": float(
        (
            diagnostic_data[
                "absolute_standardized_residual"
            ] > 2
        ).mean()
    ),
    "n_abs_standardized_residual_gt_3": int(
        (
            diagnostic_data[
                "absolute_standardized_residual"
            ] > 3
        ).sum()
    ),
    "rate_abs_standardized_residual_gt_3": float(
        (
            diagnostic_data[
                "absolute_standardized_residual"
            ] > 3
        ).mean()
    ),
}])

outlier_summary.to_csv(
    DIAGNOSTIC_DIR / "residual_outlier_summary.csv",
    index=False,
)

subject_diagnostics = (
    diagnostic_data
    .groupby("subject_id", as_index=False)
    .agg(
        n_observations=("residual", "size"),
        mean_residual=("residual", "mean"),
        residual_sd=("residual", "std"),
        mean_absolute_residual=(
            "residual",
            lambda x: np.mean(np.abs(x)),
        ),
        max_absolute_standardized_residual=(
            "absolute_standardized_residual",
            "max",
        ),
        rate_abs_standardized_residual_gt_2=(
            "absolute_standardized_residual",
            lambda x: np.mean(x > 2),
        ),
    )
    .sort_values(
        "mean_absolute_residual",
        ascending=False,
    )
)

item_diagnostics = (
    diagnostic_data
    .groupby("item_id", as_index=False)
    .agg(
        n_observations=("residual", "size"),
        mean_residual=("residual", "mean"),
        residual_sd=("residual", "std"),
        mean_absolute_residual=(
            "residual",
            lambda x: np.mean(np.abs(x)),
        ),
        max_absolute_standardized_residual=(
            "absolute_standardized_residual",
            "max",
        ),
        rate_abs_standardized_residual_gt_2=(
            "absolute_standardized_residual",
            lambda x: np.mean(x > 2),
        ),
    )
    .sort_values(
        "mean_absolute_residual",
        ascending=False,
    )
)

subject_diagnostics.to_csv(
    DIAGNOSTIC_DIR / "subject_diagnostics.csv",
    index=False,
)

item_diagnostics.to_csv(
    DIAGNOSTIC_DIR / "item_diagnostics.csv",
    index=False,
)


# Résidus versus prédictions
plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=fitted,
    y=residuals,
    alpha=0.20,
    s=18,
    edgecolor=None,
)

sns.regplot(
    x=fitted,
    y=residuals,
    scatter=False,
    lowess=True,
    color="red",
    line_kws={"linewidth": 2},
)

plt.axhline(
    0,
    color="black",
    linestyle="--",
)

plt.xlabel("Valeur ajustée")
plt.ylabel("Résidu")
plt.title("Résidus du modèle linéaire final")

save_figure(
    "final_model_residuals_vs_fitted.png"
)


# Distribution
plt.figure(figsize=(10, 7))

sns.histplot(
    residuals,
    bins=50,
    kde=True,
)

plt.axvline(
    0,
    color="black",
    linestyle="--",
)

plt.xlabel("Résidu")
plt.title("Distribution des résidus")

save_figure(
    "final_model_residual_distribution.png"
)


# QQ-plot
plt.figure(figsize=(9, 9))

stats.probplot(
    residuals,
    dist="norm",
    plot=plt,
)

plt.title("QQ-plot des résidus")

save_figure(
    "final_model_residual_qqplot.png"
)


# Résidus par valeur ajustée, hexbin
plt.figure(figsize=(10, 7))

plt.hexbin(
    fitted,
    residuals,
    gridsize=45,
    cmap="viridis",
    mincnt=1,
)

plt.colorbar(
    label="Nombre d'observations"
)

plt.axhline(
    0,
    color="red",
    linestyle="--",
)

plt.xlabel("Valeur ajustée")
plt.ylabel("Résidu")
plt.title("Densité des résidus")

save_figure(
    "final_model_residual_hexbin.png"
)


# =============================================================================
# STABILITÉ LEAVE-ONE-SUBJECT-OUT
# =============================================================================

section("STABILITÉ DES COEFFICIENTS")

full_coefficients = dict(
    zip(
        fixed_effects["parameter"],
        fixed_effects["estimate"],
    )
)

if RUN_SUBJECT_JACKKNIFE:
    subjects = sorted(
        data["subject_id"].unique()
    )

    if JACKKNIFE_MAX_SUBJECTS is not None:
        subjects = subjects[
            :JACKKNIFE_MAX_SUBJECTS
        ]

    jackknife_rows = []

    for index, removed_subject in enumerate(
        subjects,
        start=1,
    ):
        print(
            f"[{index}/{len(subjects)}] "
            f"Retrait du participant "
            f"{removed_subject}"
        )

        reduced_data = data.loc[
            data["subject_id"] != removed_subject
        ].copy()

        try:
            result, optimizer = fit_linear_model(
                data=reduced_data,
                formula=FINAL_FORMULA,
                reml=True,
                display=False,
            )

            row = {
                "removed_subject":
                    removed_subject,
                "converged":
                    bool(result.converged),
                "optimizer":
                    optimizer,
            }

            for parameter, estimate in (
                result.fe_params.items()
            ):
                row[parameter] = estimate
                row[
                    parameter + "__difference_from_full"
                ] = (
                    estimate
                    - full_coefficients.get(
                        parameter,
                        np.nan,
                    )
                )

            jackknife_rows.append(row)

        except Exception as exc:
            jackknife_rows.append({
                "removed_subject":
                    removed_subject,
                "converged":
                    False,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })

    jackknife = pd.DataFrame(
        jackknife_rows
    )

    jackknife.to_csv(
        JACKKNIFE_DIR
        / "leave_one_subject_out_coefficients.csv",
        index=False,
    )

    coefficient_columns = [
        parameter
        for parameter in fixed_effects["parameter"]
        if parameter in jackknife.columns
    ]

    stability_rows = []

    for parameter in coefficient_columns:
        estimates = pd.to_numeric(
            jackknife[parameter],
            errors="coerce",
        )

        full_estimate = full_coefficients[
            parameter
        ]

        stability_rows.append({
            "parameter": parameter,
            "full_estimate": full_estimate,
            "jackknife_mean": estimates.mean(),
            "jackknife_sd": estimates.std(ddof=1),
            "jackknife_min": estimates.min(),
            "jackknife_max": estimates.max(),
            "maximum_absolute_change": (
                estimates - full_estimate
            ).abs().max(),
            "sign_change_detected": bool(
                (
                    np.sign(estimates.dropna())
                    != np.sign(full_estimate)
                ).any()
            ),
            "n_successful_models":
                estimates.notna().sum(),
        })

    stability = pd.DataFrame(
        stability_rows
    )

    stability.to_csv(
        JACKKNIFE_DIR
        / "jackknife_stability_summary.csv",
        index=False,
    )

    plot_parameters = [
        "item_entropy_z",
        "subject_mean_models_z",
        "models_within_subject_z",
    ]

    available_parameters = [
        parameter
        for parameter in plot_parameters
        if parameter in jackknife.columns
    ]

    if available_parameters:
        long_jackknife = jackknife.melt(
            id_vars=["removed_subject"],
            value_vars=available_parameters,
            var_name="parameter",
            value_name="estimate",
        )

        plt.figure(figsize=(12, 7))

        sns.boxplot(
            data=long_jackknife,
            x="parameter",
            y="estimate",
        )

        for position, parameter in enumerate(
            available_parameters
        ):
            plt.scatter(
                position,
                full_coefficients[parameter],
                color="red",
                marker="D",
                s=80,
                label=(
                    "Estimation complète"
                    if position == 0
                    else None
                ),
            )

        plt.axhline(
            0,
            color="black",
            linestyle="--",
        )

        plt.ylabel("Coefficient")
        plt.xlabel("")
        plt.title(
            "Stabilité leave-one-subject-out"
        )
        plt.legend()

        save_figure(
            "jackknife_cognitive_coefficients.png"
        )

else:
    print(
        "Jackknife non exécuté. "
        "Mettre RUN_SUBJECT_JACKKNIFE=True "
        "pour le lancer."
    )


# =============================================================================
# CALIBRATION MÉTACOGNITIVE DESCRIPTIVE
# =============================================================================

section("CALIBRATION MÉTACOGNITIVE")

data["confidence_bin_10"] = pd.cut(
    data["confidence"],
    bins=[
        -0.001,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        99.999,
        100,
    ],
    labels=[
        "0-10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81-90",
        "91-99",
        "100",
    ],
    include_lowest=True,
)

calibration_bins = (
    data
    .groupby(
        "confidence_bin_10",
        observed=False,
        as_index=False,
    )
    .agg(
        n_observations=(
            "is_correct_binary",
            "size",
        ),
        mean_confidence=(
            "confidence_probability",
            "mean",
        ),
        observed_accuracy=(
            "is_correct_binary",
            "mean",
        ),
    )
)

calibration_bins["calibration_gap"] = (
    calibration_bins["mean_confidence"]
    - calibration_bins["observed_accuracy"]
)

calibration_bins.to_csv(
    CALIBRATION_DIR
    / "calibration_by_confidence_bin.csv",
    index=False,
)

overall_brier = brier_score_loss(
    data["is_correct_binary"],
    data["confidence_probability"],
)

overall_calibration_bias = (
    data["confidence_probability"].mean()
    - data["is_correct_binary"].mean()
)

overall_absolute_calibration_error = (
    np.average(
        np.abs(
            calibration_bins[
                "calibration_gap"
            ]
        ),
        weights=calibration_bins[
            "n_observations"
        ],
    )
)

overall_summary = pd.DataFrame([{
    "n_observations": len(data),
    "mean_confidence_probability": (
        data["confidence_probability"].mean()
    ),
    "observed_accuracy": (
        data["is_correct_binary"].mean()
    ),
    "calibration_bias_confidence_minus_accuracy":
        overall_calibration_bias,
    "absolute_calibration_error_weighted":
        overall_absolute_calibration_error,
    "brier_score": overall_brier,
}])

overall_summary.to_csv(
    CALIBRATION_DIR
    / "overall_metacognitive_calibration.csv",
    index=False,
)

print(overall_summary.to_string(index=False))


# Diagramme de calibration
plot_calibration = calibration_bins.loc[
    calibration_bins["n_observations"] > 0
].copy()

plt.figure(figsize=(9, 9))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Calibration parfaite",
)

plt.plot(
    plot_calibration["mean_confidence"],
    plot_calibration["observed_accuracy"],
    marker="o",
    linewidth=2,
    label="Calibration observée",
)

for _, row in plot_calibration.iterrows():
    plt.annotate(
        str(row["confidence_bin_10"]),
        (
            row["mean_confidence"],
            row["observed_accuracy"],
        ),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points",
    )

plt.xlim(0, 1.02)
plt.ylim(0, 1.02)
plt.xlabel("Confiance moyenne")
plt.ylabel("Proportion correcte")
plt.title("Diagramme de calibration métacognitive")
plt.legend()

save_figure(
    "metacognitive_calibration_curve.png"
)


# Distribution confiance correcte/incorrecte
plt.figure(figsize=(11, 7))

sns.kdeplot(
    data=data,
    x="confidence",
    hue="is_correct_binary",
    common_norm=False,
    fill=True,
    alpha=0.30,
)

plt.xlabel("Confiance")
plt.title(
    "Distribution de la confiance selon l'exactitude"
)

save_figure(
    "confidence_by_accuracy_distribution.png"
)


# =============================================================================
# CALIBRATION PAR PARTICIPANT
# =============================================================================

section("CALIBRATION PAR PARTICIPANT")

subject_rows = []

for subject_id, subject_data in data.groupby(
    "subject_id"
):
    y_true = subject_data[
        "is_correct_binary"
    ].to_numpy()

    y_score = subject_data[
        "confidence_probability"
    ].to_numpy()

    n_correct = int(y_true.sum())
    n_incorrect = int(
        len(y_true) - n_correct
    )

    if (
        n_correct >= MIN_CLASS_COUNT_FOR_AUC
        and n_incorrect >= MIN_CLASS_COUNT_FOR_AUC
    ):
        auc = roc_auc_score(
            y_true,
            y_score,
        )
    else:
        auc = np.nan

    correct_confidence = subject_data.loc[
        subject_data["is_correct_binary"] == 1,
        "confidence",
    ].mean()

    incorrect_confidence = subject_data.loc[
        subject_data["is_correct_binary"] == 0,
        "confidence",
    ].mean()

    subject_rows.append({
        "subject_id": subject_id,
        "condition":
            subject_data["condition"].iloc[0],
        "n_trials": len(subject_data),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "accuracy": y_true.mean(),
        "mean_confidence_probability":
            y_score.mean(),
        "calibration_bias":
            y_score.mean() - y_true.mean(),
        "brier_score":
            brier_score_loss(y_true, y_score),
        "type2_auc": auc,
        "mean_confidence_correct":
            correct_confidence,
        "mean_confidence_incorrect":
            incorrect_confidence,
        "confidence_discrimination":
            correct_confidence
            - incorrect_confidence,
    })

subject_calibration = pd.DataFrame(
    subject_rows
)

subject_calibration.to_csv(
    CALIBRATION_DIR
    / "subject_metacognitive_calibration.csv",
    index=False,
)

subject_summary = pd.DataFrame([
    {
        "metric": column,
        "mean": subject_calibration[
            column
        ].mean(),
        "median": subject_calibration[
            column
        ].median(),
        "standard_deviation":
            subject_calibration[
                column
            ].std(ddof=1),
        "minimum": subject_calibration[
            column
        ].min(),
        "maximum": subject_calibration[
            column
        ].max(),
        "n_available": subject_calibration[
            column
        ].notna().sum(),
    }
    for column in [
        "accuracy",
        "mean_confidence_probability",
        "calibration_bias",
        "brier_score",
        "type2_auc",
        "confidence_discrimination",
    ]
])

subject_summary.to_csv(
    CALIBRATION_DIR
    / "subject_calibration_summary.csv",
    index=False,
)

plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=subject_calibration,
    x="accuracy",
    y="mean_confidence_probability",
    hue="condition",
    s=70,
    alpha=0.75,
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
)

plt.xlabel("Précision du participant")
plt.ylabel("Confiance moyenne")
plt.title("Calibration moyenne par participant")

save_figure(
    "subject_mean_calibration.png"
)


plt.figure(figsize=(10, 7))

sns.histplot(
    data=subject_calibration,
    x="type2_auc",
    hue="condition",
    bins=20,
    element="step",
    common_norm=False,
)

plt.axvline(
    0.5,
    color="black",
    linestyle="--",
)

plt.xlabel("AUC métacognitive de type 2")
plt.title(
    "Discrimination métacognitive individuelle"
)

save_figure(
    "subject_type2_auc_distribution.png"
)


# =============================================================================
# GLMM LOGISTIQUE DE L'EXACTITUDE
# =============================================================================

section(
    "MODÈLE LOGISTIQUE MIXTE : "
    "EXACTITUDE SELON LA CONFIANCE"
)

calibration_model = (
    BinomialBayesMixedGLM.from_formula(
        formula=CALIBRATION_FORMULA,
        vc_formulas=VC_FORMULA_BINARY,
        data=data,
        fe_p=2.0,
        vcp_p=0.5,
    )
)

calibration_attempts = [
    {
        "fit_method": "BFGS",
        "scale_fe": True,
        "minim_opts": {
            "maxiter": 10000,
            "gtol": 1e-5,
        },
    },
    {
        "fit_method": "BFGS",
        "scale_fe": True,
        "minim_opts": {
            "maxiter": 15000,
            "gtol": 1e-4,
        },
    },
]

calibration_result = None
calibration_attempt_rows = []

for attempt_number, attempt in enumerate(
    calibration_attempts,
    start=1,
):
    print(
        f"Tentative calibration "
        f"{attempt_number}"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        candidate = calibration_model.fit_vb(
            fit_method=attempt["fit_method"],
            scale_fe=attempt["scale_fe"],
            minim_opts=attempt["minim_opts"],
            verbose=False,
        )

    retvals = (
        getattr(
            candidate,
            "optim_retvals",
            {},
        )
        or {}
    )

    success = bool(
        retvals.get("success", False)
    )

    gradient = retvals.get("jac")

    if gradient is not None:
        maximum_gradient = np.max(
            np.abs(
                np.asarray(
                    gradient,
                    dtype=float,
                )
            )
        )
    else:
        maximum_gradient = np.nan

    calibration_attempt_rows.append({
        "attempt": attempt_number,
        "success": success,
        "objective": retvals.get(
            "fun",
            np.nan,
        ),
        "iterations": retvals.get(
            "nit",
            np.nan,
        ),
        "maximum_absolute_gradient":
            maximum_gradient,
        "message": retvals.get(
            "message",
            "",
        ),
        "warnings": " | ".join(
            str(warning.message)
            for warning in caught
        ),
    })

    calibration_result = candidate

    if success:
        break

calibration_attempt_table = pd.DataFrame(
    calibration_attempt_rows
)

calibration_attempt_table.to_csv(
    CALIBRATION_DIR
    / "calibration_model_optimization.csv",
    index=False,
)

if calibration_result is None:
    raise RuntimeError(
        "Échec du modèle de calibration."
    )

print(calibration_result.summary())

with open(
    CALIBRATION_DIR
    / "accuracy_confidence_glmm_summary.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(
        str(calibration_result.summary())
    )

calibration_fixed = (
    posterior_binary_fixed_effects(
        calibration_result
    )
)

calibration_fixed.to_csv(
    CALIBRATION_DIR
    / "accuracy_confidence_glmm_fixed_effects.csv",
    index=False,
)


# =============================================================================
# FIGURE DE L'EFFET D'ENTROPIE
# =============================================================================

section("FIGURE DE L'EFFET D'ENTROPIE")

coefficient_dictionary = dict(
    zip(
        fixed_effects["parameter"],
        fixed_effects["estimate"],
    )
)

entropy_grid = np.linspace(
    data["item_entropy_z"].quantile(0.01),
    data["item_entropy_z"].quantile(0.99),
    100,
)

prediction_rows = []

for condition in ["Neutral", "Standard"]:
    condition_effect = (
        coefficient_dictionary.get(
            "C(condition, "
            "Treatment(reference='Neutral'))"
            "[T.Standard]",
            0,
        )
        if condition == "Standard"
        else 0
    )

    for entropy_value in entropy_grid:
        prediction = (
            coefficient_dictionary["Intercept"]
            + condition_effect
            + coefficient_dictionary[
                "item_entropy_z"
            ] * entropy_value
        )

        prediction_rows.append({
            "condition": condition,
            "item_entropy_z": entropy_value,
            "predicted_confidence": prediction,
        })

entropy_predictions = pd.DataFrame(
    prediction_rows
)

entropy_predictions.to_csv(
    TABLE_DIR
    / "entropy_adjusted_predictions.csv",
    index=False,
)

plt.figure(figsize=(10, 7))

sns.lineplot(
    data=entropy_predictions,
    x="item_entropy_z",
    y="predicted_confidence",
    hue="condition",
    linewidth=3,
)

plt.xlabel("Entropie de l'item standardisée")
plt.ylabel("Confiance prédite")
plt.title(
    "Association ajustée entre entropie et confiance"
)

save_figure(
    "adjusted_entropy_confidence.png"
)


# =============================================================================
# TABLEAU FINAL DES RÉSULTATS PRINCIPAUX
# =============================================================================

section("TABLEAU FINAL")

final_results_rows = []

for _, row in fixed_effects.iterrows():
    final_results_rows.append({
        "analysis": "Linear mixed model",
        "outcome": "Confidence 0-100",
        "parameter": row["parameter"],
        "estimate": row["estimate"],
        "standard_error_or_posterior_sd":
            row["standard_error"],
        "lower_95": row["ci_95_lower"],
        "upper_95": row["ci_95_upper"],
        "transformed_estimate": np.nan,
        "p_value": row["p_value"],
    })

for _, row in calibration_fixed.iterrows():
    final_results_rows.append({
        "analysis":
            "Logistic mixed calibration model",
        "outcome": "Correct response",
        "parameter": row["parameter"],
        "estimate":
            row["posterior_mean_log_odds"],
        "standard_error_or_posterior_sd":
            row["posterior_sd"],
        "lower_95":
            row[
                "credible_95_lower_log_odds"
            ],
        "upper_95":
            row[
                "credible_95_upper_log_odds"
            ],
        "transformed_estimate":
            row["odds_ratio"],
        "p_value": np.nan,
    })

final_results_table = pd.DataFrame(
    final_results_rows
)

final_results_table.to_csv(
    TABLE_DIR / "final_results_table.csv",
    index=False,
)


# =============================================================================
# MÉTADONNÉES
# =============================================================================

results_metadata = {
    "data_file": str(DATA_FILE),
    "output_directory": str(OUTPUT_DIR),
    "n_observations": int(len(data)),
    "n_subjects": int(
        data["subject_id"].nunique()
    ),
    "n_items": int(
        data["item_id"].nunique()
    ),
    "linear_formula": FINAL_FORMULA,
    "linear_optimizer": final_optimizer,
    "linear_converged": bool(
        final_result.converged
    ),
    "prediction_type_for_residuals":
        prediction_type,
    "jackknife_executed":
        RUN_SUBJECT_JACKKNIFE,
    "calibration_formula":
        CALIBRATION_FORMULA,
    "overall_brier_score":
        safe_float(overall_brier),
    "overall_calibration_bias":
        safe_float(overall_calibration_bias),
    "python_version":
        sys.version.split()[0],
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
    "statsmodels_version":
        statsmodels.__version__,
}

with open(
    OUTPUT_DIR / "analysis_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results_metadata,
        file,
        indent=4,
        ensure_ascii=False,
    )


# =============================================================================
# TERMINÉ
# =============================================================================

section("TERMINÉ")

print("Résultats :", OUTPUT_DIR)

print("\nDiagnostics principaux :")
print(
    DIAGNOSTIC_DIR
    / "residual_normality_summary.csv"
)
print(
    DIAGNOSTIC_DIR
    / "residual_outlier_summary.csv"
)
print(
    DIAGNOSTIC_DIR
    / "subject_diagnostics.csv"
)
print(
    DIAGNOSTIC_DIR
    / "item_diagnostics.csv"
)

print("\nCalibration :")
print(
    CALIBRATION_DIR
    / "overall_metacognitive_calibration.csv"
)
print(
    CALIBRATION_DIR
    / "subject_calibration_summary.csv"
)
print(
    CALIBRATION_DIR
    / "accuracy_confidence_glmm_fixed_effects.csv"
)
print(
    CALIBRATION_DIR
    / "calibration_model_optimization.csv"
)

print("\nTableaux :")
print(
    TABLE_DIR
    / "final_linear_fixed_effects.csv"
)
print(
    TABLE_DIR
    / "final_linear_variance_components.csv"
)
print(
    TABLE_DIR
    / "final_results_table.csv"
)

if not RUN_SUBJECT_JACKKNIFE:
    print(
        "\nLe jackknife n'a pas été exécuté. "
        "Après avoir vérifié le premier lancement, "
        "mettre RUN_SUBJECT_JACKKNIFE=True."
    )
