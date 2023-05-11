import torch
import json
import matplotlib.pyplot as plt
import filename_util

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


def get_usable_tensor(filename: str):
    return torch.flatten(torch.load(filename)).cpu()


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        name = ed["name"]
        plot_histogram(get_usable_tensor(filename_util.raw_pred_fn(name)),
                       ed["histogram"]["bins"],
                       tuple(ed["histogram"]["range"]),
                       ed["title"],
                       filename_util.histogram_fn(name))


if __name__ == "__main__":
    main()
