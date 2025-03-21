from csml_ensemble import *
from scipy.special import softmax
import os


tqdm.pandas()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU")
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

 

login(
    new_session=False,  # Won’t request token if one is already saved on machine
    write_permission=True,  # Requires a token with write permission
    token=os.environ["HF_TOKEN"],
)

dataset = load_dataset("Vulnerability-Detection/test_set_llm_predictions")
df: pd.DataFrame = dataset["train"].to_pandas()  # type: ignore
X = df.drop(
    ["vulnerable", "__index_level_0__"],
    axis=1,
    errors="ignore",
)
y = df["vulnerable"]

print(X)

cwes = ["672", "119", "682", "754"]
models = ["codebert", "codeT5"]
results = []

for cwe in cwes:
    for model in models:
        folder = load_model(cwe=cwe, model_name=model)  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(
            folder,
            local_files_only=True,
        )
        merged_model = AutoModelForSequenceClassification.from_pretrained(
            folder
        ).to(device)
        merged_model.to(device)
        
        # Evaluate the model
        logits = X[[f"{model}_{cwe}_0", f"{model}_{cwe}_1"]].values
        true_labels = y.values

        # Softmax
        preds = softmax(logits, axis=1)
        preds = np.argmax(preds, axis=1)

        # Print F2
        results.append((
            accuracy_score(true_labels, preds),
            precision_score(true_labels, preds),
            recall_score(true_labels, preds),
            fbeta_score(true_labels, preds, beta=1),
            fbeta_score(true_labels, preds, beta=2),
            geometric_mean_score(true_labels, preds),
            roc_auc_score(true_labels, preds),
            cwe,
            model
        ))

# Print results
results = pd.DataFrame(results, columns=pd.Index([
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f2",
    "gmean",
    "roc_auc",
    "cwe",
    "model"
]))

print(results)
results.sort_values(["model", "cwe"], ascending=True).to_csv("fulltestindividualresults.csv")
