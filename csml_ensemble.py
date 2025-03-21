import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from optspace import MODELS
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from huggingface_hub import snapshot_download
import os
from datasets import DatasetDict, load_dataset, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    accuracy_score,
    roc_auc_score,
)
from imblearn.metrics import geometric_mean_score
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from huggingface_hub import login
from sklearn.metrics import make_scorer
import shap


import warnings

warnings.filterwarnings("ignore")


def load_model(cwe: int, model_name: str):
    repo_id = f"Vulnerability-Detection/cwe{cwe}-{model_name}"
    os.makedirs(f"./kaggle/working/{cwe}_{model_name}/", exist_ok=True)
    folder = snapshot_download(
        repo_id=repo_id,
        local_dir=f"./kaggle/working/{cwe}_{model_name}/",
    )
    return folder


def collect_predictions(X, pipeline):
    # The pipe line contains a tokenizer and a model
    tokenizer = pipeline[0]
    model = pipeline[1]
    tokenize = lambda func: tokenizer(
        func, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    tokenized_text = X.progress_map(tokenize)

    def process_with_model(tokenized_input):
        outputs = model(**tokenized_input)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy().flatten()
        return logits

    predictions = tokenized_text.progress_map(process_with_model)
    return np.array(predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="all", help="Model to use [all, t5, codebert]"
    )
    parser.add_argument("--inference", action="store_true", help="Use inference mode")
    args = parser.parse_args()

    tqdm.pandas()
    login(
        new_session=False,  # Won’t request token if one is already saved on machine
        write_permission=True,  # Requires a token with write permission
        token=os.environ["HF_TOKEN"],
    )

    inference = args.inference
    if torch.cuda.is_available() and not inference:
        device = torch.device("cuda")
    else:
        print("GPU is not available, using CPU")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    cwes = ["672", "119", "682", "754"]
    models = ["codeT5", "codebert"]
    pipelines = []

    for cwe in cwes:
        for model in models:
            folder = load_model(cwe=cwe, model_name=model)  # type: ignore
            if torch.cuda.is_available():
                tokenizer = AutoTokenizer.from_pretrained(
                    folder,
                    local_files_only=True,
                )
                merged_model = AutoModelForSequenceClassification.from_pretrained(
                    folder
                ).to(device)
                merged_model.to(device)
            else:
                merged_model = None
                tokenizer = None
            pipelines.append([tokenizer, merged_model, model, cwe])

    labels = []
    test_df = pd.read_csv("./kaggle/input/vulnerabilities-dataset/test_all.csv")
    train_df = pd.read_csv("./kaggle/input/vulnerabilities-dataset/train_all.csv")

    for pipeline in pipelines:
        model, cwe = pipeline[2], pipeline[3]
        labels.extend([f"{model}_{cwe}_0", f"{model}_{cwe}_1"])

    try:
        dataset = load_dataset("Vulnerability-Detection/llm_predictions")
        df_test: pd.DataFrame = dataset["test"].to_pandas()  # type: ignore
        df_train: pd.DataFrame = dataset["train"].to_pandas()  # type: ignore

        # Test
        X_test = df_test.drop(
            ["vulnerable", "__index_level_0__"],
            axis=1,
            errors="ignore",
        )
        y_test = df_test["vulnerable"]

        # Train
        X_train = df_train.drop(
            ["vulnerable", "__index_level_0__"],
            axis=1,
            errors="ignore",
        )
        y_train = df_train["vulnerable"]

    except Exception as e:
        print(e)

        def make_predictions(input):
            print("Making predictions")
            predictions = []
            for pipeline in pipelines:
                predictions.append(collect_predictions(input["func"], pipeline))
            predictions = np.hstack([np.vstack(row) for row in np.array(predictions)])
            X_test = pd.DataFrame(predictions, columns=pd.Index(labels))
            y_test = pd.Series(input["vulnerable"])
            return X_test, y_test

        X_train, y_train = make_predictions(train_df)
        X_test, y_test = make_predictions(test_df)

        X_train.to_csv("X_train.csv")
        X_test.to_csv("X_test.csv")
        y_train.to_csv("y_train.csv")
        y_test.to_csv("y_test.csv")

        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(pd.concat([X_train, y_train], axis=1)),
                "test": Dataset.from_pandas(pd.concat([X_test, y_test], axis=1)),
            }
        )
        dataset.push_to_hub("Vulnerability-Detection/llm_predictions")

    cvf = StratifiedKFold(n_splits=5)
    results = []
    beta = 2

    selected_model = args.model
    if selected_model == "CodeT5":
        X_train = X_train.drop(
            [col for col in X_train.columns if "codebert" in col.lower()], axis=1
        )
        X_test = X_test.drop(
            [col for col in X_test.columns if "codebert" in col.lower()], axis=1
        )
    elif selected_model == "CodeBERT":
        X_train = X_train.drop(
            [col for col in X_train.columns if "t5" in col.lower()], axis=1
        )
        X_test = X_test.drop(
            [col for col in X_test.columns if "t5" in col.lower()], axis=1
        )

    assert selected_model in ["Combined", "CodeT5", "CodeBERT"]
    for label, setup in MODELS.items():
        search_space = setup["search_spaces"]
        estimator = setup["estimator"]
        f2_scoring = make_scorer(fbeta_score, beta=beta)

        model = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_space,
            cv=5,
            scoring=f2_scoring,
            refit=True,
            n_jobs=12,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fbeta = fbeta_score(y_test, y_pred, beta=beta)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print(f"{label} scored F{beta} {fbeta}, Precision {precision}, Recall {recall}")
        results.append([label, fbeta, precision, recall])

        # SHAP Analysis
        try:
            pred = model.predict(X_test)
            if label == "SVC" or label == "BalancedBagging":
                explainer = shap.KernelExplainer(
                    model.best_estimator_.predict_proba, X_train.sample(100)
                )
            else:
                explainer = shap.TreeExplainer(model.best_estimator_)
            shap_values = explainer(X_test)
            plt.figure(figsize=(15, 8))
            class_to_show = 1
            shap.plots.beeswarm(shap_values[:, :, class_to_show], show=False)
            plt.title(
                f"SHAP Analysis for {label} using {selected_model}, F2 Score: {fbeta:.2f}"
            )
            plt.tight_layout()
            os.makedirs(f"shap/{selected_model}", exist_ok=True)
            plt.savefig(f"shap/{selected_model}/shap_{label}.png", dpi=400)
            plt.close()
        except Exception as e:
            print(e)
            continue

    # Assuming your dataframe is called 'df'
    # Melt the dataframe to create a long format suitable for Seaborn
    results = pd.DataFrame(
        results,
        columns=pd.Index(["Fold", "Algorithm", f"F{beta}", "Precision", "Recall"]),
    )
    results.to_csv(f"results_{selected_model}.csv")
    df_melted = pd.melt(
        results,
        id_vars=["Fold", "Algorithm"],
        value_vars=[f"F{beta}", "Precision", "Recall"],
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Algorithm", y="Value", hue="Metric", data=df_melted)
    plt.title("ML Experiment Results")
    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"results_{selected_model}.png", dpi=500)
