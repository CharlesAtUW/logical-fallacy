import subprocess
import json
import sys

DEFAULT_TRAIN_DETAILS_FILENAME = "training_details/train_electra_small_mnli_on_logic.json"

def perform_training(details: dict, print_stdout=True, print_stderr=False):
     print(f"training model at {details['threshold_testing_args']['--model']} on module {details['module_name']}:")
     eval_title = details["title"]
     print(eval_title)

     command_args = ["python",
                    details["module_name"],
                    "--tokenizer", "google/electra-large-discriminator"]
     
     if "threshold_testing_args" in details:
          for arg, value in details["threshold_testing_args"].items():
               command_args += [arg, value]

     process = subprocess.Popen(command_args,
                                stdout=None if print_stdout else subprocess.DEVNULL,
                                stderr=None if print_stderr else subprocess.DEVNULL,
                                shell=False)
     process.communicate()


def do_model_training(eval_details_filename: str):
    all_eval_details = None
    with open(eval_details_filename, "r") as json_file:
        all_eval_details = json.load(json_file)
    
    for ed in all_eval_details:
        perform_training(ed, print_stderr=True)

if __name__ == "__main__":
    eval_file = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_TRAIN_DETAILS_FILENAME
    do_model_training(eval_file)
