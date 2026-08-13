#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

N3_FILE = (
    BASE_DIR
    / "cognitive_mixed_model_E1"
    / "cognitive_model_fixed_effects_REML.csv"
)

N20_FILE = (
    BASE_DIR
    / "cognitive_mixed_model_E1_n20"
    / "cognitive_model_fixed_effects_REML.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "mreasoner_robustness_E1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "cognitive_coefficients_n3_vs_n20.csv"
)


if not N3_FILE.exists():
    raise FileNotFoundError(
        f"Fichier n3 absent : {N3_FILE}"
    )

if not N20_FILE.exists():
    raise FileNotFoundError(
        f"Fichier n20 absent : {N20_FILE}"
    )


n3 = pd.read_csv(N3_FILE)
n20 = pd.read_csv(N20_FILE)

required_columns = [
    "parameter",
    "estimate",
    "standard_error",
    "p_value",
    "ci_95_lower",
    "ci_95_upper",
]

for name, frame in [
    ("n3", n3),
    ("n20", n20),
]:
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: colonnes absentes : "
            + ", ".join(missing)
        )


n3 = n3[required_columns].rename(
    columns={
        column: f"{column}_n3"
        for column in required_columns
        if column != "parameter"
    }
)

n20 = n20[required_columns].rename(
    columns={
        column: f"{column}_n20"
        for column in required_columns
        if column != "parameter"
    }
)

comparison = n3.merge(
    n20,
    on="parameter",
    how="outer",
    validate="one_to_one",
)

comparison["absolute_change"] = (
    comparison["estimate_n20"]
    - comparison["estimate_n3"]
)

comparison["absolute_change_magnitude"] = (
    comparison["absolute_change"].abs()
)

comparison["relative_change_percent"] = np.where(
    comparison["estimate_n3"].abs() > 1e-12,
    100
    * comparison["absolute_change"]
    / comparison["estimate_n3"].abs(),
    np.nan,
)

comparison["sign_n3"] = np.sign(
    comparison["estimate_n3"]
)

comparison["sign_n20"] = np.sign(
    comparison["estimate_n20"]
)

comparison["sign_changed"] = (
    comparison["sign_n3"]
    != comparison["sign_n20"]
)

comparison["significant_n3_0_05"] = (
    comparison["p_value_n3"] < 0.05
)

comparison["significant_n20_0_05"] = (
    comparison["p_value_n20"] < 0.05
)

comparison["significance_changed"] = (
    comparison["significant_n3_0_05"]
    != comparison["significant_n20_0_05"]
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("=" * 80)
print("COMPARAISON DES COEFFICIENTS N3 VERSUS N20")
print("=" * 80)

display_columns = [
    "parameter",
    "estimate_n3",
    "p_value_n3",
    "estimate_n20",
    "p_value_n20",
    "absolute_change",
    "sign_changed",
    "significance_changed",
]

print(
    comparison[display_columns]
    .round(6)
    .to_string(index=False)
)

print("\nFichier créé :")
print(OUTPUT_FILE)
