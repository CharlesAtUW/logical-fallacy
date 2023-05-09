import subprocess
import json

EVAL_DETAILS_FILENAME = "all_evaluation_details.json"

def perform_evaluation(details: dict, print_stdout=True, print_stderr=False):
        print(f"evaluating model at {details['model_location']} on module {details['module_name']}:")
        command_args = ["python",
                        details["module_name"],
                        "-t", "google/electra-large-discriminator",
                        "-m", details["model_location"],
                        "-ts", details["strategy"],
                        "-ds", details["strategy"],
                        "-mp", details["map"],
                        "-sm", details["metrics_save_path"],
                        "-tmin", details["threshold_min"],
                        "-tmax", details["threshold_max"],
                        "-tstep", details["threshold_step"],
                        "-sp", details["save_predictions"],
                        "-sl", details["save_labels"]]
        if details["add_nt_arg"]:
             command_args += ["-nt", "T"]
        if details["add_fm_arg"]:
             command_args += ["-fm", "T"]
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
