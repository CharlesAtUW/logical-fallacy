import os
from model_training import do_model_training
from obtain_pretrained_seq_cls import save_pretrained_sequence_classification_model, DEFAULT_MODEL

TRAIN_DETAILS_DIRNAME = "training_details"

def main():
    save_pretrained_sequence_classification_model(DEFAULT_MODEL)
    for filename in os.listdir(TRAIN_DETAILS_DIRNAME):
        train_details_filename = os.path.join(TRAIN_DETAILS_DIRNAME, filename)

        do_model_training(train_details_filename)


if __name__ == "__main__":
    main()
