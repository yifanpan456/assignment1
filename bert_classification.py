########## 1. Import required libraries ##########
import pandas as pd
import numpy as np
import re
import math
import os
import subprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import time

start_time = time.time()

# ===== New for BERT =====
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

########## 2. Define text preprocessing methods ##########
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

########## 3. Download & read data ##########
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'pytorch'
path = f'{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)

pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

########## 4. Configure parameters & Start training ##########
datafile = 'Title+Body.csv'
REPEAT = 1
out_csv_name = f'../{project}_BERT.csv'

data = pd.read_csv(datafile).fillna('')
text_col = 'text'
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

accuracies = []
precisions = []
recalls = []
f1_scores = []
auc_values = []

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

for repeated_time in range(REPEAT):
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(indices, test_size=0.2, random_state=repeated_time)

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]
    y_train = data['sentiment'].iloc[train_index]
    y_test = data['sentiment'].iloc[test_index]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    train_dataset = BERTDataset(train_text.tolist(), y_train_enc.tolist(), tokenizer)
    test_dataset = BERTDataset(test_text.tolist(), y_test_enc.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(5):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #break

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    acc = accuracy_score(y_true, y_pred)
    accuracies.append(acc)
    precisions.append(precision_score(y_true, y_pred, average='macro'))
    recalls.append(recall_score(y_true, y_pred, average='macro'))
    f1_scores.append(f1_score(y_true, y_pred, average='macro'))
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    auc_values.append(auc(fpr, tpr))

final_accuracy = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall = np.mean(recalls)
final_f1 = np.mean(f1_scores)
final_auc = np.mean(auc_values)

print("=== BERT Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

try:
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame({
    'repeated_times': [REPEAT],
    'Accuracy': [final_accuracy],
    'Precision': [final_precision],
    'Recall': [final_recall],
    'F1': [final_f1],
    'AUC': [final_auc],
    'CV_list(AUC)': [str(auc_values)]
})

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time for repeat {repeated_time + 1}: {elapsed_time:.2f} seconds")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time for repeat {repeated_time + 1}: {elapsed_time:.2f} seconds")