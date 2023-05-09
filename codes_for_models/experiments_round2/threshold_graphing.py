import pandas as pd
import matplotlib.pyplot as plt
import json

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def graph_precision_recall_curve(metrics: pd.DataFrame, additional_points: list, results_name: str):
    thresholds = metrics["threshold"].to_numpy()
    precisions = metrics["precision"].to_numpy()
    recalls = metrics["recall"].to_numpy()
    additional_precisions = [p["precision"] for p in additional_points]
    additional_recalls = [p["recall"] for p in additional_points]

    plt.figure()

    main_points = plt.scatter(precisions, recalls)
    for i, th in enumerate(thresholds):
        if i % 10 == 0:
            plt.annotate(f"{th:.2f}", (precisions[i], recalls[i]))

    additional_points = plt.scatter(additional_precisions, additional_recalls)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.legend((main_points, additional_points), ("Model's metrics", "Paper's metrics"))
    plt.suptitle(f"Precision vs. recall (for various thresholds)", fontsize=18)
    plt.title(results_name, fontsize=10)
    plt.show()


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as json_file:
        all_eval_details = json.load(json_file)

    for ed in all_eval_details:
        metrics = pd.read_csv(ed["metrics_save_path"])
        graph_precision_recall_curve(metrics, ed["additional_points"], ed["name"])


if __name__ == "__main__":
    main()
