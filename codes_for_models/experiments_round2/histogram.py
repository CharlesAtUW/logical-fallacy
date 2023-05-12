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
    plt.close()


def get_usable_tensor(filename: str, column=None):
    usable_tensor = torch.load(filename).cpu()
    if column is not None:
        usable_tensor = usable_tensor[:, column]
    return torch.flatten(usable_tensor)


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        name = ed["name"]
        if ed.get("by_fallacy_arg", "F") == "T":
            for i, fallacy in enumerate(FALLACIES):
                plot_histogram(get_usable_tensor(filename_util.raw_pred_fn(name), column=i),
                            ed["histogram"]["bins"],
                            tuple(ed["histogram"]["range"]),
                            ed["title"],
                            filename_util.histogram_by_fallacy_fn(name, fallacy.replace(" ", "_"), create_dirs=True))
        else:
            plot_histogram(get_usable_tensor(filename_util.raw_pred_fn(name)),
                        ed["histogram"]["bins"],
                        tuple(ed["histogram"]["range"]),
                        ed["title"],
                        filename_util.histogram_fn(name))


if __name__ == "__main__":
    main()
