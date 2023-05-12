import torch
import json
import matplotlib.pyplot as plt
import filename_util

from logicedu import FALLACIES
from sklearn.calibration import CalibrationDisplay
from histogram import get_usable_tensor

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def plot_calibration_curve(raw_predictions: torch.Tensor, labels: torch.Tensor, save_path: str):
    y_prob = torch.sigmoid(raw_predictions)
    
    plt.figure()
    CalibrationDisplay.from_predictions(labels, y_prob, n_bins=20)
    plt.savefig(save_path)
    plt.close()


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        name = ed["name"]
        if ed.get("by_fallacy_arg", "F") == "T":
            for i, fallacy in enumerate(FALLACIES):
                plot_calibration_curve(get_usable_tensor(filename_util.raw_pred_fn(name), column=i),
                                    get_usable_tensor(filename_util.labels_fn(name), column=i),
                                    filename_util.calibration_by_fallacy_fn(name, fallacy.replace(" ", "_"), create_dirs=True))
        else:
            plot_calibration_curve(get_usable_tensor(filename_util.raw_pred_fn(name)),
                                get_usable_tensor(filename_util.labels_fn(name)),
                                filename_util.calibration_fn(name))


if __name__ == "__main__":
    main()
