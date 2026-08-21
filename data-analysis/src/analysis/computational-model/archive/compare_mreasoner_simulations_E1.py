#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare les comptages MReasoner obtenus avec 3, 10 et 20 simulations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "mreasoner_robustness_E1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    3: BASE_DIR.parent / "mental_models_count_E1.csv",
    10: BASE_DIR / "mental_models_count_E1_n10.csv",
    20: BASE_DIR / "mental_models_count_E1_n20.csv",
}

REQUIRED = [
    "subject_id",
    "task",
    "premise_1",
    "premise_2",
    "number_models_generated",
    "std_models_generated",
    "minimum_models_generated",
    "maximum_models_generated",
    "n_samples",
]

sns.set_theme(style="whitegrid", context="talk")


def task_type_from_premises(premise_1, premise_2):
    first = str(premise_1).strip().lower()
    second = str(premise_2).strip().lower()

    mapping = {
        (
            "all b are c",
            "all a are b",
        ): "MP",
        (
            "all b are c",
            "no a are c",
        ): "MT",
        (
            "all b are c",
            "all a are c",
        ): "AC",
        (
            "all b are c",
            "no a are b",
        ): "DA",
    }

    return mapping.get(
        (first, second),
        np.nan,
    )


frames = []

for n_simulations, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier absent pour {n_simulations} "
            f"simulations : {path}"
        )

    frame = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name}: colonnes absentes : "
            + ", ".join(missing)
        )

    frame["subject_id"] = (
        frame["subject_id"].astype(str)
    )

    frame["task_type"] = [
        task_type_from_premises(
            first,
            second,
        )
        for first, second in zip(
            frame["premise_1"],
            frame["premise_2"],
        )
    ]

    if frame["task_type"].isna().any():
        raise ValueError(
            f"{path.name}: certaines tâches "
            "ne peuvent pas être identifiées."
        )

    frame["simulation_setting"] = (
        n_simulations
    )

    frames.append(frame)

long_data = pd.concat(
    frames,
    ignore_index=True,
)

long_data.to_csv(
    OUTPUT_DIR
    / "mreasoner_all_simulation_settings.csv",
    index=False,
)

summary = (
    long_data
    .groupby(
        ["simulation_setting", "task_type"],
        as_index=False,
    )
    .agg(
        n_combinations=(
            "number_models_generated",
            "size",
        ),
        mean_models=(
            "number_models_generated",
            "mean",
        ),
        sd_between_combinations=(
            "number_models_generated",
            "std",
        ),
        mean_simulation_sd=(
            "std_models_generated",
            "mean",
        ),
        median_simulation_sd=(
            "std_models_generated",
            "median",
        ),
        mean_range=(
            "maximum_models_generated",
            lambda x: np.nan,
        ),
    )
)

range_summary = (
    long_data
    .assign(
        simulation_range=(
            long_data["maximum_models_generated"]
            - long_data["minimum_models_generated"]
        )
    )
    .groupby(
        ["simulation_setting", "task_type"],
        as_index=False,
    )
    .agg(
        mean_range=("simulation_range", "mean"),
        median_range=("simulation_range", "median"),
        zero_sd_rate=(
            "std_models_generated",
            lambda x: np.mean(x == 0),
        ),
    )
)

summary = summary.drop(
    columns=["mean_range"]
).merge(
    range_summary,
    on=["simulation_setting", "task_type"],
    how="left",
)

summary.to_csv(
    OUTPUT_DIR
    / "mreasoner_stability_summary.csv",
    index=False,
)


wide = (
    long_data
    .pivot_table(
        index=["subject_id", "task_type"],
        columns="simulation_setting",
        values="number_models_generated",
        aggfunc="first",
    )
    .reset_index()
)

wide.columns = [
    (
        f"models_n{column}"
        if isinstance(column, int)
        else column
    )
    for column in wide.columns
]

for first, second in [
    (3, 10),
    (3, 20),
    (10, 20),
]:
    first_column = f"models_n{first}"
    second_column = f"models_n{second}"

    wide[
        f"difference_n{second}_minus_n{first}"
    ] = (
        wide[second_column]
        - wide[first_column]
    )

wide.to_csv(
    OUTPUT_DIR
    / "mreasoner_combination_comparison.csv",
    index=False,
)


comparison_rows = []

for first, second in [
    (3, 10),
    (3, 20),
    (10, 20),
]:
    first_column = f"models_n{first}"
    second_column = f"models_n{second}"

    valid = wide[[
        first_column, second_column
    ]].dropna()

    differences = (
        valid[second_column]
        - valid[first_column]
    )

    comparison_rows.append({
        "comparison":
            f"{first}_vs_{second}",
        "n_combinations": len(valid),
        "pearson_correlation":
            valid[first_column].corr(
                valid[second_column],
                method="pearson",
            ),
        "spearman_correlation":
            valid[first_column].corr(
                valid[second_column],
                method="spearman",
            ),
        "mean_difference":
            differences.mean(),
        "mean_absolute_difference":
            np.abs(differences).mean(),
        "root_mean_squared_difference":
            np.sqrt(
                np.mean(differences ** 2)
            ),
        "maximum_absolute_difference":
            np.abs(differences).max(),
        "rate_absolute_difference_gt_0_25":
            np.mean(
                np.abs(differences) > 0.25
            ),
        "rate_absolute_difference_gt_0_50":
            np.mean(
                np.abs(differences) > 0.50
            ),
    })

comparison = pd.DataFrame(
    comparison_rows
)

comparison.to_csv(
    OUTPUT_DIR
    / "mreasoner_setting_comparison.csv",
    index=False,
)


plt.figure(figsize=(11, 7))

sns.boxplot(
    data=long_data,
    x="simulation_setting",
    y="std_models_generated",
)

plt.xlabel("Nombre de simulations")
plt.ylabel("Écart-type entre simulations")
plt.title(
    "Stabilité des estimations MReasoner"
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR
    / "mreasoner_simulation_sd.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


if {
    "models_n3",
    "models_n20",
}.issubset(wide.columns):
    plt.figure(figsize=(9, 9))

    sns.scatterplot(
        data=wide,
        x="models_n3",
        y="models_n20",
        hue="task_type",
        s=60,
        alpha=0.75,
    )

    minimum = min(
        wide["models_n3"].min(),
        wide["models_n20"].min(),
    )

    maximum = max(
        wide["models_n3"].max(),
        wide["models_n20"].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        color="black",
    )

    plt.xlabel("Nombre de modèles — 3 simulations")
    plt.ylabel("Nombre de modèles — 20 simulations")
    plt.title(
        "Comparaison 3 versus 20 simulations"
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR
        / "mreasoner_n3_vs_n20.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


print("Analyse terminée.")
print("Résultats :", OUTPUT_DIR)
print(comparison.to_string(index=False))
