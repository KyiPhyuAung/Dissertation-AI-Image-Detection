from pathlib import Path
import pandas as pd

from src.config import RESULTS_DIR


def main():
    rows = []

    experiment_dirs = [
        path for path in RESULTS_DIR.iterdir()
        if path.is_dir()
    ]

    for experiment_dir in experiment_dirs:
        result_file = experiment_dir / "cross_generator_results.csv"

        if not result_file.exists():
            continue

        df = pd.read_csv(result_file)
        df["experiment_folder"] = experiment_dir.name

        rows.append(df)

    if not rows:
        print("No experiment result files found.")
        return

    summary_df = pd.concat(rows, ignore_index=True)

    output_path = RESULTS_DIR / "master_cross_generator_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print("=" * 70)
    print("Master Results Summary")
    print("=" * 70)
    print(summary_df[[
        "model",
        "train_generator",
        "test_generator",
        "accuracy",
        "fake_recall",
        "fake_f1"
    ]])
    print("=" * 70)
    print(f"Saved master summary to: {output_path}")


if __name__ == "__main__":
    main()