import os

def create_dirs_if_nonexistent(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def get_filename(base_dir: str, *names, create_dirs=False):
    filename = os.path.join(base_dir, *names)
    if create_dirs:
        create_dirs_if_nonexistent(filename)
    return filename


def raw_pred_fn(name: str, create_dirs=False):
    return get_filename("raw_predictions", name, create_dirs=create_dirs)


def labels_fn(name: str, create_dirs=False):
    return get_filename("raw_labels", name, create_dirs=create_dirs)


def plots_fn(name: str, create_dirs=False):
    return get_filename("plots", name + ".png", create_dirs=create_dirs)


def plots_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_filename("plots", name, fallacy + ".png", create_dirs=create_dirs)


def plots_by_fallacy_dn(name: str, create_dirs=False):
    return get_filename("plots", name, create_dirs=create_dirs)


def metrics_fn(name: str, create_dirs=False):
    return get_filename("metrics", name + ".csv", create_dirs=create_dirs)


def metrics_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_filename("metrics", name, fallacy + ".csv", create_dirs=create_dirs)


def metrics_by_fallacy_dn(name: str, create_dirs=False):
    return get_filename("metrics", name, create_dirs=create_dirs)


def histogram_fn(name: str, create_dirs=False):
    return get_filename("distributions", name + ".png", create_dirs=create_dirs)


def histogram_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_filename("distributions", name, fallacy + ".png", create_dirs=create_dirs)


def histogram_by_fallacy_dn(name: str, create_dirs=False):
    return get_filename("distributions", name, create_dirs=create_dirs)


def calibration_fn(name: str, create_dirs=False):
    return get_filename("calibration_plots", name + ".png", create_dirs=create_dirs)


def calibration_by_fallacy_fn(name: str, fallacy: str, create_dirs=False):
    return get_filename("calibration_plots", name, fallacy + ".png", create_dirs=create_dirs)


def calibration_by_fallacy_dn(name: str, create_dirs=False):
    return get_filename("calibration_plots", name, create_dirs=create_dirs)
