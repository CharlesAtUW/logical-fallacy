import os
from model_training import do_model_training
from obtain_pretrained_seq_cls import save_pretrained_sequence_classification_model

TRAIN_DETAILS_DIRNAME = "training_details"
FINETUNE_START_KEYWORD = "finetune"

def starts_with(name: str, start_keyword: str):
    return len(name) >= len(start_keyword) and name[:len(start_keyword)] == start_keyword

def main():
    f"""
    In order to always train before finetuning, filenames in {TRAIN_DETAILS_DIRNAME} that start with
    the string {FINETUNE_START_KEYWORD} will be processed after all others.
    """
    
    save_pretrained_sequence_classification_model("howey/electra-small-mnli")
    save_pretrained_sequence_classification_model("howey/electra-base-mnli")

    all_filenames = os.listdir(TRAIN_DETAILS_DIRNAME)
    training_filenames = [f for f in all_filenames if not starts_with(f, FINETUNE_START_KEYWORD)]
    finetune_filenames = [f for f in all_filenames if starts_with(f, FINETUNE_START_KEYWORD)]

    for filename in training_filenames:
        train_details_filename = os.path.join(TRAIN_DETAILS_DIRNAME, filename)
        print(train_details_filename)
        do_model_training(train_details_filename)

    for filename in finetune_filenames:
        train_details_filename = os.path.join(TRAIN_DETAILS_DIRNAME, filename)
        print(train_details_filename)
        do_model_training(train_details_filename)


if __name__ == "__main__":
    main()
