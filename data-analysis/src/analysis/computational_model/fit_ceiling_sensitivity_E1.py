#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset_analysis_E1.csv"
OUTPUT_DIR = BASE_DIR / "ceiling_sensitivity_E1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORMULA = (
    "confidence ~ "
    "C(condition, Treatment(reference='Neutral')) "
    "+ sequence_c10 "
    "+ subject_accuracy_z "
    "+ item_entropy_z "
    "+ subject_mean_models_z "
    "+ models_within_subject_z"
)

VC_FORMULA = {
    "item": "0 + C(item_id)",
    "subject": "0 + C(subject_id)",
}

REQUIRED = [
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


def standardize(series):
    series = pd.to_numeric(series, errors="coerce")
    return (series - series.mean()) / series.std(ddof=1)


def fit_mixed_model(data, reml):
    model_data = data.copy()
    model_data["_global_group"] = 1

    model = smf.mixedlm(
        FORMULA,
        model_data,
        groups=model_data["_global_group"],
        re_formula="0",
        vc_formula=VC_FORMULA,
    )

    for optimizer in ["lbfgs", "bfgs", "cg", "powell"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(
                    reml=reml,
                    method=optimizer,
                    maxiter=3000,
                    full_output=True,
                    disp=False,
                )

            if result.converged:
                return result, optimizer, caught

        except Exception as exc:
            print(f"Échec {optimizer}: {exc}")

    raise RuntimeError("Aucun ajustement convergé.")


def fixed_effects(result):
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


if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

data = pd.read_csv(DATA_FILE)

missing = [column for column in REQUIRED if column not in data.columns]
if missing:
    raise ValueError("Colonnes absentes : " + ", ".join(missing))

if "analysis_complete" in data.columns:
    mask = (
        data["analysis_complete"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    data = data.loc[mask].copy()

for column in [
    "confidence",
    "sequence",
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(subset=REQUIRED).copy()

data["subject_id"] = data["subject_id"].astype(str)
data["item_id"] = data["item_id"].astype(str)
data["condition"] = data["condition"].astype(str).str.strip()

data["sequence_c10"] = (
    data["sequence"] - data["sequence"].mean()
) / 10

for variable in [
    "subject_accuracy",
    "item_entropy",
    "subject_mean_models",
    "models_within_subject",
]:
    data[variable + "_z"] = standardize(data[variable])

data["at_ceiling"] = (data["confidence"] == 100).astype(int)

ceiling_summary = pd.DataFrame([{
    "n_total": len(data),
    "n_at_ceiling": int(data["at_ceiling"].sum()),
    "ceiling_rate": data["at_ceiling"].mean(),
    "n_below_ceiling": int((data["confidence"] < 100).sum()),
}])

ceiling_summary.to_csv(
    OUTPUT_DIR / "ceiling_summary.csv",
    index=False,
)

print("\nDistribution du plafond :")
print(ceiling_summary.to_string(index=False))


# Analyse principale sans les valeurs égales à 100
below_ceiling = data.loc[data["confidence"] < 100].copy()

print("\nLignes conservées :", len(below_ceiling))
print("Participants :", below_ceiling["subject_id"].nunique())
print("Items :", below_ceiling["item_id"].nunique())

result_ml, optimizer_ml, warnings_ml = fit_mixed_model(
    below_ceiling,
    reml=False,
)

result_reml, optimizer_reml, warnings_reml = fit_mixed_model(
    below_ceiling,
    reml=True,
)

fixed_ml = fixed_effects(result_ml)
fixed_reml = fixed_effects(result_reml)

fixed_ml.to_csv(
    OUTPUT_DIR / "below_ceiling_fixed_effects_ML.csv",
    index=False,
)

fixed_reml.to_csv(
    OUTPUT_DIR / "below_ceiling_fixed_effects_REML.csv",
    index=False,
)

with open(
    OUTPUT_DIR / "below_ceiling_model_ML.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(str(result_ml.summary()))

with open(
    OUTPUT_DIR / "below_ceiling_model_REML.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(str(result_reml.summary()))

print("\nMODÈLE SOUS LE PLAFOND — REML")
print(result_reml.summary())

print("\nOptimiseur ML :", optimizer_ml)
print("Optimiseur REML :", optimizer_reml)
print("\nRésultats enregistrés dans :", OUTPUT_DIR)
