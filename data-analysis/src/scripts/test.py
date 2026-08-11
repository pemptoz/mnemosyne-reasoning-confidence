from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "E4_syllogismData_full.csv"
)

CCOBRA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_ccobra_E2_ref.csv"
)


raw = pd.read_csv(
    RAW_FILE
)

ccobra = pd.read_csv(
    CCOBRA_FILE
)



def normalize_id(series):
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


raw["subject_id"] = normalize_id(
    raw["sona_id"]
)

ccobra["subject_id"] = normalize_id(
    ccobra["id"]
)

raw["sequence_normalized"] = pd.to_numeric(
    raw["trial_num"],
    errors="coerce",
)

ccobra["sequence_normalized"] = pd.to_numeric(
    ccobra["sequence"],
    errors="coerce",
)

# Ensemble des clés présentes dans CCOBRA.
ccobra_keys = set(
    zip(
        ccobra["subject_id"],
        ccobra["sequence_normalized"],
    )
)

raw["included_in_ccobra"] = [
    (subject_id, sequence) in ccobra_keys
    for subject_id, sequence in zip(
        raw["subject_id"],
        raw["sequence_normalized"],
    )
]

# On analyse seulement les lignes dont la correction est disponible.
usable = raw.dropna(
    subset=[
        "correct_int",
        "correct_ref",
    ]
).copy()

print("=" * 70)
print("ESSAIS CONSERVÉS ET EXCLUS")
print("=" * 70)

summary = (
    usable
    .groupby(
        "included_in_ccobra"
    )
    .agg(
        number_of_trials=(
            "trial_num",
            "size",
        ),
        number_of_subjects=(
            "subject_id",
            "nunique",
        ),
        intuitive_accuracy=(
            "correct_int",
            "mean",
        ),
        reflective_accuracy=(
            "correct_ref",
            "mean",
        ),
    )
)

summary["intuitive_accuracy"] *= 100
summary["reflective_accuracy"] *= 100

print(summary)

columns_to_check = [
    "condition",
    "type",
    "validity",
    "believability",
    "conflict",
]

for column in columns_to_check:
    if column not in usable.columns:
        continue

    print("\n" + "=" * 70)
    print("RÉPARTITION PAR", column.upper())
    print("=" * 70)

    counts = pd.crosstab(
        usable[column],
        usable["included_in_ccobra"],
        margins=True,
    )

    print(counts)

    print("\nPrécision intuitive :")

    print(
        usable.groupby(
            [
                "included_in_ccobra",
                column,
            ],
            dropna=False,
        )["correct_int"]
        .mean()
        .mul(100)
    )

    print("\nPrécision réfléchie :")

    print(
        usable.groupby(
            [
                "included_in_ccobra",
                column,
            ],
            dropna=False,
        )["correct_ref"]
        .mean()
        .mul(100)
    )

print("\n" + "=" * 70)
print("PARTICIPANTS EXCLUS")
print("=" * 70)

raw_subjects = set(
    usable["subject_id"].dropna()
)

ccobra_subjects = set(
    ccobra["subject_id"].dropna()
)

excluded_subjects = sorted(
    raw_subjects - ccobra_subjects
)

print(excluded_subjects)

print("\n" + "=" * 70)
print("TAUX DE CONSERVATION PAR PARTICIPANT")
print("=" * 70)

participant_summary = (
    usable
    .groupby(
        "subject_id",
        as_index=False,
    )
    .agg(
        raw_trials=(
            "trial_num",
            "size",
        ),
        included_trials=(
            "included_in_ccobra",
            "sum",
        ),
        intuitive_accuracy=(
            "correct_int",
            "mean",
        ),
        reflective_accuracy=(
            "correct_ref",
            "mean",
        ),
    )
)

participant_summary["retention_rate"] = (
    100
    * participant_summary["included_trials"]
    / participant_summary["raw_trials"]
)

participant_summary["intuitive_accuracy"] *= 100
participant_summary["reflective_accuracy"] *= 100

print(
    participant_summary
    .sort_values(
        by=[
            "retention_rate",
            "subject_id",
        ]
    )
    .to_string(index=False)
)
