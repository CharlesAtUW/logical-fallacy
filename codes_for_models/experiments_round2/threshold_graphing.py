import pandas as pd
import matplotlib.pyplot as plt
import json
import filename_util

from typing import List

from logicedu import FALLACIES

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def graph_precision_recall_curve(eval_details: dict, metrics: pd.DataFrame,
                                 additional_points: list, save_path: str, show: bool = False):
    results_title = eval_details["title"]

    thresholds = metrics["threshold"].to_numpy()
    precisions = metrics["precision"].to_numpy()
    recalls = metrics["recall"].to_numpy()
    additional_precisions: List[float] = [p["precision"] for p in additional_points]
    additional_recalls: List[float] = [p["recall"] for p in additional_points]

    plt.figure()

    main_scatter = plt.scatter(recalls, precisions)
    for i, th in enumerate(thresholds):
        if i % 10 == 0:
            plt.annotate(f"{th:.2f}", (recalls[i], precisions[i]))

    additional_scatter = plt.scatter(additional_recalls, additional_precisions)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend((main_scatter, additional_scatter), ("Model's metrics", "Paper's metrics"))
    plt.suptitle(f"Precision vs. recall (for various thresholds)", fontsize=18)
    plt.title(results_title, fontsize=10)
    
    if show:
        plt.show()
    plt.savefig(save_path)
    plt.close()


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as json_file:
        all_eval_details = json.load(json_file)

    for ed in all_eval_details:
        if ed.get("by_fallacy_arg", "F") == "T":
            for fallacy in FALLACIES:
                metrics = pd.read_csv(filename_util.metrics_by_fallacy_fn(ed["name"], fallacy.replace(" ", "_"), create_dirs=True))
                graph_precision_recall_curve(ed, metrics, ed.get("additional_points", {}).get(fallacy, []),
                                             filename_util.plots_by_fallacy_fn(ed["name"], fallacy.replace(" ", "_"), create_dirs=True))
        else:
            metrics = pd.read_csv(filename_util.metrics_fn(ed["name"]))
            graph_precision_recall_curve(ed, metrics, ed.get("additional_points", []), filename_util.plots_fn(ed["name"]))


if __name__ == "__main__":
    main()
