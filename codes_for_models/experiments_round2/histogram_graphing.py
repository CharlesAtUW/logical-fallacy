import torch
import json
import matplotlib.pyplot as plt
import filename_util

from logicedu import FALLACIES

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

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


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        eval_name = ed["name"]
        title = ed["title"]
        num_bins = ed["histogram"]["bins"]
        histogram_range = tuple(ed["histogram"]["range"])

        if ed.get("by_fallacy_arg", "F") == "T":
            for i, fallacy in enumerate(FALLACIES):
                converted_fallacy = fallacy_to_name_in_files(fallacy)
                plot_histogram(get_usable_tensor(filename_util.raw_pred_fn(eval_name), column=i),
                            num_bins,
                            histogram_range,
                            add_fallacy_to_title(title, fallacy),
                            filename_util.histogram_by_fallacy_fn(eval_name, converted_fallacy, create_dirs=True))
        else:
            plot_histogram(get_usable_tensor(filename_util.raw_pred_fn(eval_name)),
                        num_bins,
                        histogram_range,
                        title,
                        filename_util.histogram_fn(eval_name))


if __name__ == "__main__":
    main()
