from transformers import AutoModelForSequenceClassification, AdamW
from logicedu import get_logger, MNLIDataset, train, eval1, pretty_print_scores, save_metrics_csv, FALLACIES
import argparse
import pandas as pd
import torch
import os
from library import eval_classwise, eval_and_store, convert_to_multilabel

torch.manual_seed(0)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger = get_logger()
    logger.info("device = %s", device)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", help="tokenizer path")
    parser.add_argument("-m", "--model", help="model path")
    parser.add_argument("-w", "--weight", help="Weight of entailment loss")
    parser.add_argument("-s", "--savepath", help="Path to save logicclimate model")
    parser.add_argument("-mp", "--map", help="Map labels to this category")
    parser.add_argument("-ts", "--train_strat", help="Strategy number for training")
    parser.add_argument("-ds", "--dev_strat", help="Strategy number for development and testing")
    parser.add_argument("-f", "--finetune", help="Set this flag if you want to finetune the model on LogicClimate",
                        default='F')
    parser.add_argument("-c", "--classwise_savepath", help="Path to store classwise results")
    parser.add_argument("-sr", "--result_path", help="Path to store results on dev set")
    parser.add_argument("-sm", "--metrics_path", help="Path to store metrics on dev set")
    parser.add_argument("-tmin", "--threshold_min", help="Minimum threshold to try on evals")
    parser.add_argument("-tmax", "--threshold_max", help="Maximum threshold (excluded) to try on evals")
    parser.add_argument("-tstep", "--threshold_step", help="Increment thresholds by this value")
    parser.add_argument("-fm", "--finetuned_model", help="Set this flag if the model was finetuned on LogicClimate",
                        default='F')
    parser.add_argument("-sp", "--save_predictions", help="Save raw predictions to this file")
    parser.add_argument("-sl", "--save_labels", help="Save raw labels to this file")
    parser.add_argument("-ed", "--eval_dataset", help="Dataset to use when running evals. Can be \"train\", \"dev\", or \"test\".",
                        default="test")
    parser.add_argument("-bf", "--by_fallacy", help="Set to true to separate evals by fallacy", default="F")
    parser.add_argument("-esf", "--early_stopping_ft", help="When finetuning, stop when an epoch is worse than the previous",
                        default="T")
    parser.add_argument("-nef", "--num_epochs_ft", help="Number of epochs when finetuning", default="10")
    parser.add_argument("-pra", "--precision_recall_averaging", help="Averaging method for calculating precision and recall", default="samples")
    args = parser.parse_args()
    # word_bank = pickle.load('../../data/word_bank.pkl')
    logger.info(args)
    logger.info("initializing model")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    logger.info("creating dataset")
    fallacy_train = pd.read_csv('../../data/climate_train_mh.csv')
    fallacy_dev = pd.read_csv('../../data/climate_dev_mh.csv')
    fallacy_test = pd.read_csv('../../data/climate_test_mh.csv')
    fallacy_train['logical_fallacies'] = fallacy_train['logical_fallacies'].apply(eval)
    fallacy_dev['logical_fallacies'] = fallacy_dev['logical_fallacies'].apply(eval)
    fallacy_test['logical_fallacies'] = fallacy_test['logical_fallacies'].apply(eval)
    if args.finetune == 'F' and args.finetuned_model == 'F':
        fallacy_test = pd.concat([fallacy_train, fallacy_dev, fallacy_test])
    test_datasets = {"train": fallacy_train, "dev": fallacy_dev, "test": fallacy_test}
    fallacy_ds = MNLIDataset(args.tokenizer, fallacy_train, fallacy_dev, 'logical_fallacies', args.map, test_datasets[args.eval_dataset],
                             fallacy=True, train_strat=int(args.train_strat), test_strat=int(args.dev_strat),
                             multilabel=True)
    model.resize_token_embeddings(len(fallacy_ds.tokenizer))
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    if args.finetune == 'T':
        logger.info("starting training")
        train(model, fallacy_ds, optimizer, logger, args.savepath, device, ratio=1, epochs=int(args.num_epochs_ft),
              positive_weight=float(args.weight) if args.weight is not None else None,
              early_stopping=args.early_stopping_ft == "T")

        model = AutoModelForSequenceClassification.from_pretrained(args.savepath, num_labels=3)
        model.to(device)
    model.eval()
    logger.info("starting testing")
    _, _, test_loader = fallacy_ds.get_data_loaders()
    scores = eval1(model, test_loader, logger, device,
                   threshold_min=float(args.threshold_min),
                   threshold_max=float(args.threshold_max),
                   threshold_step=float(args.threshold_step),
                   predictions_filename=args.save_predictions,
                   labels_filename=args.save_labels,
                   by_fallacy=args.by_fallacy == "T",
                   pr_averaging=args.precision_recall_averaging)
    #logger.info("micro f1: %f macro f1:%f precision: %f recall: %f exact match %f", scores[4], scores[5], scores[1],
    #            scores[2], scores[3])
    if args.by_fallacy == "T":
        for fallacy in FALLACIES:
            print(fallacy)
            pretty_print_scores(scores[fallacy])
            save_metrics_csv(scores[fallacy], os.path.join(args.metrics_path, fallacy.replace(" ", "_") + ".csv"))
    else:
        pretty_print_scores(scores)
        save_metrics_csv(scores, args.metrics_path)

    if args.classwise_savepath is not None:
        classwise_scores = eval_classwise(model, test_loader, logger, fallacy_ds.unique_labels, device)
        df = pd.DataFrame.from_records(classwise_scores, columns=['Fallacy Name', 'Precision', 'Recall', 'F1',
                                                                  'Number of Positive Labels for this Class in Test Set'
                                                                  ])
        logger.info(classwise_scores)
        df.to_csv(args.classwise_savepath)
    if args.result_path is not None:
        logger.info("Generating results on Dev Set")
        df = eval_and_store(fallacy_ds, model, logger, device)
        df = convert_to_multilabel(df)
        df.to_csv(args.result_path)
