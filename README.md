
# Bug Report Classification Tool

This project implements two methods for automatically classifying whether a GitHub bug report is performance-related or not:

- **Baseline**: TF-IDF + Naive Bayes (`br_classification.py`)
- **Proposed Method**: BERT-based classifier (`bert_classification.py`)

## 📁 Project Structure

```
.
├── br_classification.py         # Baseline method
├── bert_classification.py       # BERT-based method
├── pytorch.csv                  # Dataset example (replaceable)
├── tensorflow.csv
├── keras.csv
├── incubator-mxnet.csv
├── caffe.csv
├── requirements.txt             # Python package dependencies
├── manual.pdf                   # How to use this tool
├── replication.pdf              # How to replicate reported results
└── README.md                    # Project overview (this file)
```

## ⚙️ Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

If you're running for the first time, also download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

## 🏁 How to Run

1. Choose the dataset by editing this line in either script:

```python
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'keras'
path = f'{project}.csv'
```

2. Run the baseline model:

```bash
python br_classification.py
```

3. Run the BERT-based model:

```bash
python bert_classification.py
```

## 📊 Output

The scripts will print evaluation metrics (Accuracy, Precision, Recall, F1, AUC) to the console and optionally save them to CSV files for comparison.

## 📌 Notes

- The BERT model uses `bert-base-uncased` from HuggingFace and runs in feature-extraction mode (frozen encoder).
- For better performance in low-resource scenarios (e.g., caffe.csv), consider using data augmentation techniques.
- All datasets should be placed in the same directory as the Python scripts.

## 📄 Documentation

- `manual.pdf`: Step-by-step guide to using the tool
- `replication.pdf`: Exact instructions to replicate the results
- `requirements.txt`: Required packages and versions

## 🔗 Author

This tool was developed as part of the Intelligent Software Engineering coursework (Lab 1).
