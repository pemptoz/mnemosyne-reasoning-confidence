#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINAL_DIR = BASE_DIR / "final_analysis_E1_n20"
REPORT_FILE = FINAL_DIR / "final_report_E1.md"

FINAL_TABLE_DIR = FINAL_DIR / "tables"
CALIBRATION_DIR = FINAL_DIR / "calibration"
DIAGNOSTIC_DIR = FINAL_DIR / "diagnostics"

COGNITIVE_DIR = (
    BASE_DIR / "cognitive_mixed_model_E1_n20"
)

SENSITIVITY_DIR = (
    BASE_DIR / "sensitivity_mixed_model_E1"
)

CEILING_DIR = (
    BASE_DIR
    / "ceiling_logistic_mixed_model_E1"
)


def read_csv(path, required=True):
    if path.exists():
        return pd.read_csv(path)

    if required:
        raise FileNotFoundError(path)

    return None


def format_number(value, digits=3):
    try:
        value = float(value)

        if not np.isfinite(value):
            return "NA"

        if (
            value != 0
            and abs(value) < 0.001
        ):
            return f"{value:.2e}"

        return f"{value:.{digits}f}"

    except (TypeError, ValueError):
        return "NA"


def find_parameter(table, text):
    match = table.loc[
        table["parameter"]
        .astype(str)
        .str.contains(
            text,
            case=False,
            regex=False,
        )
    ]

    if len(match) == 0:
        return None

    return match.iloc[0]


linear_fixed = read_csv(
    FINAL_TABLE_DIR
    / "final_linear_fixed_effects.csv"
)

variance = read_csv(
    FINAL_TABLE_DIR
    / "final_linear_variance_components.csv"
)

calibration = read_csv(
    CALIBRATION_DIR
    / "overall_metacognitive_calibration.csv"
)

subject_calibration = read_csv(
    CALIBRATION_DIR
    / "subject_calibration_summary.csv"
)

calibration_glmm = read_csv(
    CALIBRATION_DIR
    / "accuracy_confidence_glmm_fixed_effects.csv"
)

residual_normality = read_csv(
    DIAGNOSTIC_DIR
    / "residual_normality_summary.csv"
)

residual_outliers = read_csv(
    DIAGNOSTIC_DIR
    / "residual_outlier_summary.csv"
)

lr_tests = read_csv(
    COGNITIVE_DIR
    / "likelihood_ratio_tests.csv",
    required=False,
)

drop_one = read_csv(
    SENSITIVITY_DIR
    / "drop_one_tests.csv",
    required=False,
)

ceiling_fixed = read_csv(
    CEILING_DIR
    / "ceiling_logistic_fixed_effects.csv",
    required=False,
)

metadata_path = (
    FINAL_DIR / "analysis_metadata.json"
)

if metadata_path.exists():
    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)
else:
    metadata = {}


condition = find_parameter(
    linear_fixed,
    "[T.Standard]",
)

sequence = find_parameter(
    linear_fixed,
    "sequence_c10",
)

accuracy = find_parameter(
    linear_fixed,
    "subject_accuracy_z",
)

entropy = find_parameter(
    linear_fixed,
    "item_entropy_z",
)

mean_models = find_parameter(
    linear_fixed,
    "subject_mean_models_z",
)

within_models = find_parameter(
    linear_fixed,
    "models_within_subject_z",
)

confidence_accuracy = find_parameter(
    calibration_glmm,
    "confidence_z",
)


lines = []

lines.append("# Analyse de l’expérience E1")
lines.append("")
lines.append("## Méthodes")
lines.append("")
lines.append("### Participants et observations")
lines.append("")
lines.append(
    f"L’analyse portait sur "
    f"{metadata.get('n_observations', 'NA')} essais, "
    f"provenant de "
    f"{metadata.get('n_subjects', 'NA')} participants "
    f"et de {metadata.get('n_items', 'NA')} items."
)
lines.append("")

lines.append("### Variable dépendante")
lines.append("")
lines.append(
    "La variable dépendante principale était la confiance, "
    "mesurée sur une échelle allant de 0 à 100."
)
lines.append("")

lines.append("### Prédicteurs")
lines.append("")
lines.append(
    "Le modèle final incluait la condition expérimentale, "
    "la position de l’essai, la précision moyenne du "
    "participant, l’entropie empirique de l’item, le nombre "
    "moyen de modèles mentaux du participant et la composante "
    "intra-individuelle du nombre de modèles."
)
lines.append("")
lines.append(
    "Les prédicteurs continus cognitifs furent standardisés. "
    "La position de l’essai fut centrée et exprimée par "
    "tranches de dix essais."
)
lines.append("")

lines.append("### Modèle statistique")
lines.append("")
lines.append(
    "Un modèle linéaire mixte fut ajusté avec des intercepts "
    "aléatoires croisés pour les participants et les items. "
    "Les comparaisons entre structures d’effets fixes furent "
    "effectuées par maximum de vraisemblance, tandis que le "
    "modèle final fut estimé par maximum de vraisemblance "
    "restreint."
)
lines.append("")

lines.append("### Analyses de sensibilité")
lines.append("")
lines.append(
    "Les analyses de sensibilité remplacèrent la validité par "
    "le type de tâche, exclurent les réponses de confiance "
    "égales à 100 et modélisèrent séparément la probabilité "
    "d’utiliser cette borne supérieure."
)
lines.append("")

lines.append("### Calibration métacognitive")
lines.append("")
lines.append(
    "La calibration fut décrite en comparant la confiance "
    "exprimée à la proportion de réponses correctes. Le score "
    "de Brier, le biais de calibration, la discrimination "
    "correct–incorrect et l’AUC métacognitive individuelle "
    "furent également calculés."
)
lines.append("")

lines.append("## Résultats")
lines.append("")

lines.append("### Modèle linéaire final")
lines.append("")

for label, row in [
    ("Condition Standard", condition),
    ("Séquence", sequence),
    ("Précision du participant", accuracy),
    ("Entropie de l’item", entropy),
    ("Nombre moyen de modèles", mean_models),
    ("Composante intra-individuelle des modèles", within_models),
]:
    if row is None:
        continue

    lines.append(
        f"- **{label}** : "
        f"β = {format_number(row['estimate'])}, "
        f"SE = {format_number(row['standard_error'])}, "
        f"IC 95 % "
        f"[{format_number(row['ci_95_lower'])}, "
        f"{format_number(row['ci_95_upper'])}], "
        f"p = {format_number(row['p_value'])}."
    )

lines.append("")

lines.append("### Composantes de variance")
lines.append("")

for _, row in variance.iterrows():
    lines.append(
        f"- {row['component']} : "
        f"variance = "
        f"{format_number(row['variance'])}, "
        f"écart-type = "
        f"{format_number(row['standard_deviation'])}, "
        f"proportion = "
        f"{format_number(100 * row['proportion'], 1)} %."
    )

lines.append("")

lines.append("### Calibration métacognitive")
lines.append("")

calibration_row = calibration.iloc[0]

lines.append(
    f"La confiance moyenne exprimée était de "
    f"{format_number(100 * calibration_row['mean_confidence_probability'], 1)} %, "
    f"alors que la précision observée était de "
    f"{format_number(100 * calibration_row['observed_accuracy'], 1)} %. "
    f"Le biais global de calibration était de "
    f"{format_number(100 * calibration_row['calibration_bias_confidence_minus_accuracy'], 1)} "
    f"points de pourcentage et le score de Brier était de "
    f"{format_number(calibration_row['brier_score'])}."
)
lines.append("")

if confidence_accuracy is not None:
    lines.append(
        f"Dans le modèle logistique mixte, une augmentation "
        f"d’un écart-type de confiance était associée à un "
        f"odds ratio de réponse correcte de "
        f"{format_number(confidence_accuracy['odds_ratio'])}, "
        f"IC crédible à 95 % "
        f"[{format_number(confidence_accuracy['credible_95_lower_odds_ratio'])}, "
        f"{format_number(confidence_accuracy['credible_95_upper_odds_ratio'])}]."
    )
    lines.append("")

lines.append("### Diagnostics")
lines.append("")

normality_row = residual_normality.iloc[0]
outlier_row = residual_outliers.iloc[0]

lines.append(
    f"Les résidus présentaient une asymétrie de "
    f"{format_number(normality_row['skewness'])} et un excès "
    f"de kurtosis de "
    f"{format_number(normality_row['excess_kurtosis'])}. "
    f"La proportion de résidus standardisés dépassant "
    f"|2| était de "
    f"{format_number(100 * outlier_row['rate_abs_standardized_residual_gt_2'], 2)} %, "
    f"et celle dépassant |3| de "
    f"{format_number(100 * outlier_row['rate_abs_standardized_residual_gt_3'], 2)} %."
)
lines.append("")

lines.append("### Conclusion générale")
lines.append("")
lines.append(
    "L’entropie empirique des items constituait le prédicteur "
    "le plus robuste de la confiance. Les participants étaient "
    "moins confiants lorsque les réponses suscitées par un item "
    "étaient plus dispersées. Cet effet subsistait après prise "
    "en compte du type de tâche et après exclusion des réponses "
    "situées à la borne supérieure. Les résultats concernant "
    "le nombre de modèles mentaux étaient plus faibles et "
    "devront être réévalués avec davantage de simulations."
)
lines.append("")

lines.append("## Figures disponibles")
lines.append("")
lines.append(
    "- `figures/final_model_residuals_vs_fitted.png`"
)
lines.append(
    "- `figures/final_model_residual_qqplot.png`"
)
lines.append(
    "- `figures/metacognitive_calibration_curve.png`"
)
lines.append(
    "- `figures/subject_mean_calibration.png`"
)
lines.append(
    "- `figures/subject_type2_auc_distribution.png`"
)
lines.append(
    "- `figures/adjusted_entropy_confidence.png`"
)
lines.append("")

REPORT_FILE.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("Rapport créé :", REPORT_FILE)
