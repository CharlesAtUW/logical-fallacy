import os

def create_dirs_if_nonexistent(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def get_file(base_dir: str, *names, create_dirs=False):
    filename = os.path.join(base_dir, *names)
    if create_dirs:
        create_dirs_if_nonexistent(filename)
    return filename


def raw_pred_fn(name: str, create_dirs=False):
    return get_file("raw_predictions", name, create_dirs=create_dirs)


def labels_fn(name: str, create_dirs=False):
    return get_file("raw_labels", name, create_dirs=create_dirs)


def plots_fn(name: str, create_dirs=False):
    return get_file("plots", name + ".png", create_dirs=create_dirs)


def plots_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_file("plots", name, fallacy + ".png", create_dirs=create_dirs)


def plots_by_fallacy_dn(name: str, create_dirs=False):
    return get_file("plots", name, create_dirs=create_dirs)


def metrics_fn(name: str, create_dirs=False):
    return get_file("metrics", name + ".csv", create_dirs=create_dirs)


def metrics_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_file("metrics", name, fallacy + ".csv", create_dirs=create_dirs)


def metrics_by_fallacy_dn(name: str, create_dirs=False):
    return get_file("metrics", name, create_dirs=create_dirs)


def histogram_fn(name: str, create_dirs=False):
    return get_file("distributions", name + ".png", create_dirs=create_dirs)


def histogram_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_file("distributions", name, fallacy + ".png", create_dirs=create_dirs)


def histogram_by_fallacy_dn(name: str, create_dirs=False):
    return get_file("distributions", name, create_dirs=create_dirs)


def calibration_fn(name: str, create_dirs=False):
    return get_file("calibration_plots", name + ".png", create_dirs=create_dirs)


def calibration_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_file("calibration_plots", name, fallacy + ".png", create_dirs=create_dirs)


def calibration_by_fallacy_dn(name: str, create_dirs=False):
    return get_file("calibration_plots", name, create_dirs=create_dirs)
