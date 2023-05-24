import os
from threshold_testing import do_threshold_testing
from threshold_graphing import do_threshold_graphing
from calibration_graphing import do_calibration_graphing
from histogram_graphing import do_histogram_graphing

EVAL_DETAILS_DIRNAME = "evaluation_details"

def main():
    for filename in os.listdir(EVAL_DETAILS_DIRNAME):
        eval_details_filename = os.path.join(EVAL_DETAILS_DIRNAME, filename)

        do_threshold_testing(eval_details_filename)
        do_threshold_graphing(eval_details_filename)
        do_calibration_graphing(eval_details_filename)
        do_histogram_graphing(eval_details_filename)


if __name__ == "__main__":
    main()
