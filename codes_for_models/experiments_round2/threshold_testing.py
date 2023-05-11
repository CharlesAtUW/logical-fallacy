import subprocess
import json
import filename_util

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def perform_evaluation(details: dict, print_stdout=True, print_stderr=False):
        print(f"evaluating model at {details['model_location']} on module {details['module_name']}:")
        name = details["name"]
        print(name)

        command_args = ["python",
                        details["module_name"],
                        "-t", "google/electra-large-discriminator",
                        "-m", details["model_location"],
                        "-ts", details["strategy"],
                        "-ds", details["strategy"],
                        "-mp", details["map"],
                        "-sm", filename_util.metrics_fn(name),
                        "-tmin", details["threshold_min"],
                        "-tmax", details["threshold_max"],
                        "-tstep", details["threshold_step"],
                        "-sp", filename_util.raw_pred_fn(name),
                        "-sl", filename_util.labels_fn(name)]
        if details["add_nt_arg"]:
             command_args += ["-nt", "T"]
        if details["add_fm_arg"]:
             command_args += ["-fm", "T"]
        if details.get("ed_arg", False):
             command_args += ["-ed", details["ed_arg"]]
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
        perform_evaluation(ed)

if __name__ == "__main__":
    main()
