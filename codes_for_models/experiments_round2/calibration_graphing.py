import torch
import json
import matplotlib.pyplot as plt
import filename_util
from sklearn.calibration import CalibrationDisplay

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def plot_calibration_curve(raw_predictions: torch.Tensor, labels: torch.Tensor, save_path: str):
    y_prob = torch.sigmoid(raw_predictions)
    
    plt.figure()
    CalibrationDisplay.from_predictions(labels, y_prob, n_bins=20)
    plt.savefig(save_path)
    plt.close()


def get_usable_tensor(filename: str):
    return torch.flatten(torch.load(filename)).cpu()


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        name = ed["name"]
        plot_calibration_curve(get_usable_tensor(filename_util.raw_pred_fn(name)),
                               get_usable_tensor(filename_util.labels_fn(name)),
                               filename_util.calibration_fn(name))


if __name__ == "__main__":
    main()
