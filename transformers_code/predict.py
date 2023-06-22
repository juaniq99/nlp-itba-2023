from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from torch import cuda
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset
from tqdm import tqdm


def predict(model_path,column_name, data_df, expected=None):

    # Get text from data
    text = data_df[column_name].to_list()

    # Set tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    # Load model, set evaluation model
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    # Predict using pipeline, using truncation and padding. Run in gpu
    device = 0 if cuda.is_available() else -1
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, top_k=None, truncation=True, padding="max_length")
    
    # Make text data a HuggingFace dataset
    datasetObj = ListDataset(text)

    # Outputs a list of dicts like [[{'label': 'NEGATIVE', 'score': 0.0001223755971295759},  {'label': 'POSITIVE', 'score': 0.9998776316642761}]]
    predictions = []
    for out in tqdm(pipe(datasetObj)):
        predictions.append(out)

    # Print accuracy
    if expected is not None:
        correct = 0
        for i in range(len(predictions)):
            if predictions[i][0]["label"] == "LABEL_" + str(expected[i]):
                correct += 1
        print("Accuracy: " + str(correct/len(predictions)))

    save_predictions_to_csv(data_df, predictions, expected)

    return predictions

def save_predictions_to_csv(data_df, predictions, expected):
    # Save LABEL_0 and LABEL_1 scores in different columns
    label_0_scores = []
    label_1_scores = []
    obtained_labels = []
    for obj in predictions:
        label_preds = [0,0]
        for label_pred in obj:
            if label_pred["label"] == "LABEL_0":
                label_0_scores.append(label_pred["score"])
                label_preds[0] = label_pred["score"]
            else:
                label_1_scores.append(label_pred["score"])
                label_preds[1] = label_pred["score"]
        if(label_preds[0] > label_preds[1]):
            obtained_labels.append('0')
        else:
            obtained_labels.append('1')
            
    data_df["label_0"] = label_0_scores
    data_df["label_1"] = label_1_scores
    data_df["predicted"] = obtained_labels
    if expected is not None:
        data_df["expected"] = expected
    data_df.to_csv("predictions/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv", index=False)


# For showing progress bar on predict
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

