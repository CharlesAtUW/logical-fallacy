import subprocess
import json
import filename_util
import sys

DEFAULT_EVAL_DETAILS_FILENAME = "evaluation_details/authors_saved.json"

NON_THRESHOLD_TESTING_ARG = "T"
NON_THRESHOLD_TESTING_THRESHOLDS = {
    "--threshold_min": "0.5",
    "--threshold_max": "0.501",
    "--threshold_step": "0.01",
    "--do_not_save_metrics": "T"
}

def perform_evaluation(details: dict, print_stdout=True, print_stderr=False, non_threshold_testing=False):
    print(f"evaluating model at {details['threshold_testing_args']['--model']} on module {details['module_name']}:")
    eval_title = details["title"]
    eval_name = details["name"]
    dir_details = details["dir_details"]
    print(eval_title)
    print(eval_name)

    command_args = ["python",
                details["module_name"],
                "--tokenizer", "google/electra-large-discriminator",
                "--save_predictions", filename_util.raw_pred_fname(dir_details, eval_name),
                "--save_labels", filename_util.labels_fname(dir_details, eval_name)]

    for arg, value in details["threshold_testing_args"].items():
        if non_threshold_testing and arg in NON_THRESHOLD_TESTING_THRESHOLDS:
            command_args += [arg, NON_THRESHOLD_TESTING_THRESHOLDS[arg]]
        else:
            command_args += [arg, value]

    if details.get("split_by_fallacy", False):
        metrics_save_name = filename_util.metrics_by_fallacy_dname(dir_details, eval_name)
    else:
        metrics_save_name = filename_util.metrics_fname(dir_details, eval_name)
    command_args += ["--metrics_path", metrics_save_name]

    process = subprocess.Popen(command_args,
                            stdout=None if print_stdout else subprocess.DEVNULL,
                            stderr=None if print_stderr else subprocess.DEVNULL,
                            shell=False)
    process.communicate()


def do_threshold_testing(eval_details_filename: str, **kwargs):
    all_eval_details = None
    with open(eval_details_filename, "r") as json_file:
        all_eval_details = json.load(json_file)
    
    for ed in all_eval_details:
        perform_evaluation(ed, print_stderr=True, **kwargs)

if __name__ == "__main__":
    eval_file = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_EVAL_DETAILS_FILENAME
    non_threshold_testing = sys.argv[2] == NON_THRESHOLD_TESTING_ARG if len(sys.argv) >= 3 else False
    do_threshold_testing(eval_file, non_threshold_testing=non_threshold_testing)
