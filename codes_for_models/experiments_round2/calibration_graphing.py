import torch
import json
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def plot_calibration_curve(raw_predictions: torch.Tensor, labels: torch.Tensor, save_path: str):
    y_prob = torch.sigmoid(raw_predictions)
    
    plt.figure()
    CalibrationDisplay.from_predictions(labels, y_prob, n_bins=20)
    plt.savefig(save_path)


def get_usable_tensor(filename: str):
    return torch.flatten(torch.load(filename)).cpu()


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        plot_calibration_curve(get_usable_tensor(ed["save_predictions"]),
                               get_usable_tensor(ed["save_labels"]),
                               ed["calibration_save_path"])


if __name__ == "__main__":
    main()
