import subprocess
import json
import filename_util

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def perform_evaluation(details: dict, print_stdout=True, print_stderr=False):
        print(f"evaluating model at {details['model_location']} on module {details['module_name']}:")
        name = details["name"]
        print(name)

        metrics_save_name = filename_util.metrics_fn(name)
        command_args = ["python",
                        details["module_name"],
                        "-t", "google/electra-large-discriminator",
                        "-m", details["model_location"],
                        "-ts", details["strategy"],
                        "-ds", details["strategy"],
                        "-mp", details["map"],
                        "-tmin", details["threshold_min"],
                        "-tmax", details["threshold_max"],
                        "-tstep", details["threshold_step"],
                        "-sp", filename_util.raw_pred_fn(name),
                        "-sl", filename_util.labels_fn(name)]
        if "do_not_train_arg" in details:
             command_args += ["-nt", details["do_not_train_arg"]]
        if "finetuned_model_arg" in details:
             command_args += ["-fm", details["finetuned_model_arg"]]
        if "eval_dataset_arg" in details:
             command_args += ["-ed", details["eval_dataset_arg"]]
        if "by_fallacy_arg" in details:
             command_args += ["-bf", details["by_fallacy_arg"]]
             if details["by_fallacy_arg"] == "T":
                  metrics_save_name = filename_util.metrics_by_fallacy_dn(name)
        command_args += ["-sm", metrics_save_name]

        completed_process = subprocess.run(command_args, capture_output=True)

        if print_stdout:
            print(f"stdout:\n{completed_process.stdout.decode('utf-8')}")
        if print_stderr:
            print(f"stderr:\n{completed_process.stderr.decode('utf-8')}")


def main():
    all_eval_details = None
    with open(EVAL_DETAILS_FILENAME, "r") as json_file:
        all_eval_details = json.load(json_file)
    
    for ed in all_eval_details:
        if "by-fallacy" in ed["name"]:
            perform_evaluation(ed, print_stderr=True)

if __name__ == "__main__":
    main()
