import subprocess
import json
import filename_util
import sys

DEFAULT_EVAL_DETAILS_FILENAME = "evaluation_details/authors_saved.json"

def perform_evaluation(details: dict, print_stdout=True, print_stderr=False):
     print(f"evaluating model at {details['threshold_testing_args']['--model']} on module {details['module_name']}:")
     name = details["name"]
     print(name)

     metrics_save_name = filename_util.metrics_fn(name)
     command_args = ["python",
                    details["module_name"],
                    "--tokenizer", "google/electra-large-discriminator",
                    "--save_predictions", filename_util.raw_pred_fn(name),
                    "--save_labels", filename_util.labels_fn(name)]
     
     if "threshold_testing_args" in details:
          for arg, value in details["threshold_testing_args"].items():
               command_args += [arg, value]

     if details.get("split_by_fallacy", False):
        metrics_save_name = filename_util.metrics_by_fallacy_dn(name)
     command_args += ["--metrics_path", metrics_save_name]

     process = subprocess.Popen(command_args,
                                stdout=None if print_stdout else subprocess.DEVNULL,
                                stderr=None if print_stderr else subprocess.DEVNULL,
                                shell=False)
     process.communicate()


def do_threshold_testing(eval_details_filename: str):
    all_eval_details = None
    with open(eval_details_filename, "r") as json_file:
        all_eval_details = json.load(json_file)
    
    for ed in all_eval_details:
        perform_evaluation(ed, print_stderr=True)

if __name__ == "__main__":
    eval_file = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_EVAL_DETAILS_FILENAME
    do_threshold_testing(eval_file)
