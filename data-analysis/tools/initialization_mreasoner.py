import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIRECTORY = PROJECT_ROOT / "src" / "models"

if str(MODELS_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(MODELS_DIRECTORY),
    )

import mreasoner


def main():
    ccl_root = PROJECT_ROOT / ".ccl"
    mreasoner_root = PROJECT_ROOT / ".mreasoner"

    print("Installation de ClozureCL :", ccl_root)
    print("Sources mReasoner :", mreasoner_root)

    # Télécharge ClozureCL si .ccl/ n'existe pas encore.
    clozure = mreasoner.ClozureCL(
        str(ccl_root)
    )

    # Télécharge les sources Lisp si .mreasoner/ n'existe pas encore.
    source_directory = mreasoner.source_path(
        str(mreasoner_root)
    )

    mr = mreasoner.MReasoner(
        clozure.exec_path(),
        source_directory,
    )

    try:
        result = mr.query([
            "All B are C",
            "All A are B",
        ])

        print("Résultat du test :", result)
        print("Initialisation de mReasoner réussie.")

    finally:
        mr.terminate()


if __name__ == "__main__":
    main()
