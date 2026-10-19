import re

TEXT_FIELDS = ("problem", "method")
LIST_FIELDS = ("limitations", "datasets", "metrics", "baselines")
RESULT_FIELD = "results"
VERIFY_FIELDS = ("datasets", "metrics", "baselines", "results")  # checked against the cited text

# Two LLM calls per paper, each with its own targeted retrieval (keeps prompts small).
GROUPS = {
    "overview": ("problem", "method", "limitations"),
    "experiments": ("datasets", "metrics", "results", "baselines"),
}

# Retrieval queries used to find the passages that should contain each field.
QUERIES = {
    "problem": "problem statement research objective motivation aim of this study",
    "method": "proposed method approach model architecture methodology",
    "limitations": "limitations future work weaknesses shortcomings",
    "datasets": "study area and data description: data source, region, time period, number of observations, training and testing split",
    "metrics": "evaluation metrics accuracy RMSE MAE MAPE R2 correlation coefficient F1 score error measures",
    "results": "results performance comparison table best achieved outperforms",
    "baselines": "baseline models compared against existing methods benchmark",
}

FORMATS = {
    "problem": 'string: one or two sentences on the problem the paper addresses',
    "method": 'string: one or two sentences on the main method or model',
    "limitations": 'list of short strings: limitations the authors state',
    "datasets": 'list of strings: the DATA the study uses (source, region, time period, size), not model names or derived indices',
    "metrics": 'list of strings: evaluation metrics as written',
    "results": 'list of objects: {"method": string, "dataset": string or null, "metric": string, "value": string}. Tables may appear flattened: if you cannot tell which method or column a number belongs to, OMIT that number instead of guessing',
    "baselines": 'list of strings: models the proposed method is compared with',
}

# Small alias table so "RF" and "Random Forest" match across papers. Extend as you meet new names.
ALIASES = {
    "rf": "Random Forest", "randomforest": "Random Forest",
    "svm": "SVM", "supportvectormachine": "SVM",
    "xgb": "XGBoost", "xgboost": "XGBoost",
    "lstm": "LSTM", "cnn": "CNN", "gru": "GRU",
    "rmse": "RMSE", "rootmeansquareerror": "RMSE", "rootmeansquarederror": "RMSE",
    "mae": "MAE", "meanabsoluteerror": "MAE",
    "f1": "F1", "f1score": "F1", "fscore": "F1",
    "acc": "Accuracy", "accuracy": "Accuracy",
    "r2": "R2", "rsquared": "R2",
}


def squash(s: str) -> str:
    """Lowercase and drop everything except letters and digits, so 'F1-score' == 'f1 score'."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def words(s: str) -> str:
    """Lowercased alphanumeric words joined by single spaces: 'SPEI-3' -> 'spei 3', '0.214' -> '0 214'."""
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def contains_words(haystack_words: str, needle: str) -> bool:
    """Whole-word match, so 'ANN' does not match 'annual' and '0.31' does not match '0.031'.
    haystack_words must come from words()."""
    n = words(needle)
    return bool(n) and f" {n} " in f" {haystack_words} "


def canonical(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" .,;:")
    return ALIASES.get(squash(name), name)