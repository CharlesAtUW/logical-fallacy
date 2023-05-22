import pandas as pd
import matplotlib.pyplot as plt
import json
import filename_util
import sys

from typing import List

from histogram_graphing import fallacy_to_name_in_files, add_fallacy_to_title

from logicedu import FALLACIES
from threshold_testing import DEFAULT_EVAL_DETAILS_FILENAME

def graph_precision_recall_curve(metrics: pd.DataFrame,
                                 additional_points: list,
                                 save_path: str,
                                 results_title: str,
                                 show: bool = False):
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
    if len(additional_precisions) == 0:
        plt.legend((main_scatter,), ("Model's metrics",))
    else:
        plt.legend((main_scatter, additional_scatter), ("Model's metrics", "Paper's metrics"))
    plt.suptitle(f"Precision vs. recall (for various thresholds)", fontsize=18)
    plt.title(results_title, fontsize=10)
    
    if show:
        plt.show()
    plt.savefig(save_path)
    plt.clf()


def do_threshold_graphing(eval_details_filename: str):
    all_eval_details = None
    with open(eval_details_filename, "r") as json_file:
        all_eval_details = json.load(json_file)

    for ed in all_eval_details:
        eval_name = ed["name"]
        title = ed["title"]

        if ed.get("by_fallacy_arg", "F") == "T":
            for fallacy in FALLACIES:
                converted_fallacy = fallacy_to_name_in_files(fallacy)
                metrics = pd.read_csv(filename_util.metrics_by_fallacy_fn(eval_name, converted_fallacy, create_dirs=True))
                graph_precision_recall_curve(metrics,
                                             ed.get("additional_points", {}).get(fallacy, []),
                                             filename_util.plots_by_fallacy_fn(eval_name, converted_fallacy, create_dirs=True),
                                             add_fallacy_to_title(title, fallacy))
        else:
            metrics = pd.read_csv(filename_util.metrics_fn(eval_name))
            graph_precision_recall_curve(metrics,
                                         ed.get("additional_points", []),
                                         filename_util.plots_fn(eval_name),
                                         title)


if __name__ == "__main__":
    eval_file = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_EVAL_DETAILS_FILENAME
    do_threshold_graphing(eval_file)
