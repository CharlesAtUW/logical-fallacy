## Environment used
1. Install Python. Version 3.9.2 is used.
2. Install Conda. Steps can be found in the "Install Conda" section of https://docs.google.com/document/d/1iuG6dNRAuhOU7K2ZeLNzIaV1w2InvMGq6VazBzZKFu4/.
3. In the `codes_for_models/zeroshot` folder,
    - run `conda env create -f env.yml`.
    - run `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz --no-deps`

## Testing with `logicedu.py` and `logicclimate.py`
In the `codes_for_models/experiments_round2` directory, the `threshold_testing.py` script will test saved models on the Logic and LogicClimate datasets, to try to reproduce the metrics stated in the paper (in Tables 5 and 7, respectively). Running `python threshold_testing.py` will run these evals for various threshold values (in determining if an input has a certain fallacy or not), print metrics for each threshold, and save these metrics to CSV files.

### Other notes
- For each input sample into the model, and for each of the 13 fallacies, the model will classify if the sample has the fallacy ("entailment"), doesn't ("contradiction"), or neutral ("neutral"). None of the training examples used in `logicedu.py` have the "neutral" label for any fallacy.
  - This makes the model more like 13 binary classifiers instead of a multi-class classifier.
  - By default, `logicedu.py` can't easily change threshold values (from the "threshold of 0.5") (since for each fallacy, the model outputs three numbers, and makes the classification based on which number is higheset). To be able to change threshold values, we subtract the "contradiction" output number from the "entailment" number. As a consequence, the "neutral" class no longer gets selected.
- In the dataset used in `logicclimate.py`, for some reason, there is exactly one input sample that doesn't have 13 fallacies associated with it, but only 10. This causes an error to occur in the `eval1` function, so this one example is excluded in our testing.