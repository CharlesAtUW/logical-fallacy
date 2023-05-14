import torch
import json
import matplotlib.pyplot as plt
import filename_util

from logicedu import FALLACIES
from sklearn.calibration import CalibrationDisplay
from histogram_graphing import get_usable_tensor, fallacy_to_name_in_files

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def plot_calibration_curve(raw_predictions: torch.Tensor, labels: torch.Tensor, save_path: str, num_buckets: int):
    y_prob = torch.sigmoid(raw_predictions)
    
    plt.figure()
    CalibrationDisplay.from_predictions(labels, y_prob, n_bins=num_buckets)
    plt.savefig(save_path)
    plt.clf()


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as f:
        all_eval_details = json.load(f)

    for ed in all_eval_details:
        eval_name = ed["name"]
        predictions_filename = filename_util.raw_pred_fn(eval_name)
        labels_filename = filename_util.labels_fn(eval_name)
        num_buckets = ed["calibration"]["num_buckets"]

        if ed.get("by_fallacy_arg", "F") == "T":
            for i, fallacy in enumerate(FALLACIES):
                converted_fallacy = fallacy_to_name_in_files(fallacy)
                plot_calibration_curve(get_usable_tensor(predictions_filename, column=i),
                                    get_usable_tensor(labels_filename, column=i),
                                    filename_util.calibration_by_fallacy_fn(eval_name, converted_fallacy, create_dirs=True),
                                    num_buckets)
        else:
            plot_calibration_curve(get_usable_tensor(predictions_filename),
                                get_usable_tensor(labels_filename),
                                filename_util.calibration_fn(eval_name),
                                num_buckets)


if __name__ == "__main__":
    main()
