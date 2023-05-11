import os

def raw_pred_fn(name: str):
    return os.path.join("raw_predictions", name)


def labels_fn(name: str):
    return os.path.join("raw_labels", name)


def plots_fn(name: str):
    return os.path.join("plots", name + ".png")


def metrics_fn(name: str):
    return os.path.join("metrics", name + ".csv")


def histogram_fn(name: str):
    return os.path.join("distributions", name + ".png")


def calibration_fn(name: str):
    return os.path.join("calibration_plots", name + ".png")
