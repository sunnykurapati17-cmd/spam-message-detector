# Spam Message Detector

A Python machine-learning application that classifies SMS or email messages as **SPAM** or **HAM** (not spam).

It uses:

- TF-IDF for text feature extraction
- Multinomial Naive Bayes for classification
- Train/test split and evaluation metrics

## Dataset

Place a dataset named `spam.csv` in the project folder.

A common format is the Kaggle **SMS Spam Collection Dataset**, with columns:

- `v1`: label (`ham` or `spam`)
- `v2`: message text

The program automatically renames these columns to `label` and `message`.

You can also use a CSV with these exact columns:

```csv
label,message
ham,Hey, are we meeting tomorrow?
spam,Congratulations! You won a free prize. Click here now.
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Example:

```text
Enter message: Congratulations! You have won a free iPhone. Click here.
Prediction: SPAM
Confidence: 98.00%
```

## Files

- `app.py` — main program
- `requirements.txt` — dependencies
- `spam.csv` — add your dataset here
