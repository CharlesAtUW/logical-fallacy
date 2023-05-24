import torch
import json
import matplotlib.pyplot as plt
import filename_util
import sys

from logicedu import FALLACIES
from threshold_testing import DEFAULT_EVAL_DETAILS_FILENAME

def plot_histogram(raw_predictions: torch.Tensor,
                   num_bins: int,
                   bucket_range: tuple,
                   results_title: str,
                   save_path: str,
                   show: bool = False):
    y_prob = torch.sigmoid(raw_predictions).numpy()

    plt.figure()
    plt.hist(y_prob, bins=num_bins, range=bucket_range)

    plt.xlabel("Sigmoid value")
    plt.ylabel("Number of examples with that sigmoid value")
    plt.suptitle("Distribution of model sigmoid outputs", fontsize=18)
    plt.title(results_title, fontsize=10)

    if show:
        plt.show()
    plt.savefig(save_path)
    plt.clf()


def get_usable_tensor(filename: str, column=None):
    usable_tensor = torch.load(filename).cpu()
    if column is not None:
        usable_tensor = usable_tensor[:, column]
    return torch.flatten(usable_tensor)


def fallacy_to_name_in_files(fallacy: str):
    return fallacy.replace(" ", "_")


def add_fallacy_to_title(title: str, fallacy: str):
    return f"{title} ({fallacy})"


def do_histogram_graphing(eval_details_filename):
    all_eval_details = None
    with open(eval_details_filename, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        eval_name = ed["name"]
        dir_details = ed["dir_details"]
        title = ed["title"]
        num_bins = ed["histogram"]["bins"]
        histogram_range = tuple(ed["histogram"]["range"])

        if ed.get("split_by_fallacy", False):
            for i, fallacy in enumerate(FALLACIES):
                converted_fallacy = fallacy_to_name_in_files(fallacy)
                plot_histogram(get_usable_tensor(filename_util.raw_pred_fname(dir_details, eval_name), column=i),
                            num_bins,
                            histogram_range,
                            add_fallacy_to_title(title, fallacy),
                            filename_util.histogram_by_fallacy_fname(dir_details, eval_name, converted_fallacy, create_dirs=True))
        else:
            plot_histogram(get_usable_tensor(filename_util.raw_pred_fname(dir_details, eval_name)),
                        num_bins,
                        histogram_range,
                        title,
                        filename_util.histogram_fname(dir_details, eval_name))


if __name__ == "__main__":
    eval_file = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_EVAL_DETAILS_FILENAME
    do_histogram_graphing(eval_file)
