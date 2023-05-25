import os

from typing import List

CREATE_DIRS_DEFAULT = True

def create_dirs_if_nonexistent(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def get_filename(base_dir: str, subdirs: List[str], filename: str, create_dirs=False):
    combined_dirs = os.path.join(base_dir, *subdirs)
    full_filename = os.path.join(combined_dirs, filename)
    if create_dirs:
        create_dirs_if_nonexistent(full_filename)
    return full_filename


def raw_pred_fname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("raw_predictions", subdirs, name, create_dirs=create_dirs)


def labels_fname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("raw_labels", subdirs, name, create_dirs=create_dirs)


def plots_fname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("threshold_plots", subdirs, name + ".png", create_dirs=create_dirs)


def plots_by_fallacy_fname(subdirs: List[str], name: str, fallacy: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("threshold_plots", subdirs + [name], fallacy + ".png", create_dirs=create_dirs)


def plots_by_fallacy_dname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("threshold_plots", subdirs, name, create_dirs=create_dirs)


def metrics_fname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("metrics", subdirs, name + ".csv", create_dirs=create_dirs)


def metrics_by_fallacy_fname(subdirs: List[str], name: str, fallacy: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("metrics", subdirs + [name], fallacy + ".csv", create_dirs=create_dirs)


def metrics_by_fallacy_dname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("metrics", subdirs, name, create_dirs=create_dirs)


def histogram_fname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("distribution_histograms", subdirs, name + ".png", create_dirs=create_dirs)


def histogram_by_fallacy_fname(subdirs: List[str], name: str, fallacy: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("distribution_histograms", subdirs + [name], fallacy + ".png", create_dirs=create_dirs)


def histogram_by_fallacy_dname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("distribution_histograms", subdirs, name, create_dirs=create_dirs)


def calibration_fname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("calibration_plots", subdirs, name + ".png", create_dirs=create_dirs)


def calibration_by_fallacy_fname(subdirs: List[str], name: str, fallacy: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("calibration_plots", subdirs + [name], fallacy + ".png", create_dirs=create_dirs)


def calibration_by_fallacy_dname(subdirs: List[str], name: str, create_dirs=CREATE_DIRS_DEFAULT):
    return get_filename("calibration_plots", subdirs, name, create_dirs=create_dirs)
