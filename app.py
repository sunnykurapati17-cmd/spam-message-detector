import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "spam.csv"

def load_dataset(path=DATASET_PATH):
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except FileNotFoundError:
        raise FileNotFoundError(
            "spam.csv not found. Download a spam SMS dataset and place it in this folder."
        )

    # Common Kaggle SMS Spam Collection format: v1 = label, v2 = message
    if {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "label", "v2": "message"})
        df = df[["label", "message"]]
    elif {"label", "message"}.issubset(df.columns):
        df = df[["label", "message"]]
    else:
        raise ValueError("Dataset must contain columns named 'label' and 'message'.")

    df = df.dropna()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    return df


def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
        )),
        ("classifier", MultinomialNB()),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model training complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return model


def predict_message(model, message):
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = probabilities.max()
    return prediction, confidence


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} messages.")
    print(dataset["label"].value_counts())

    spam_model = train_model(dataset)

    print("\nSpam Message Detector")
    print("Type a message to classify it, or type 'exit' to quit.")

    while True:
        user_message = input("\nEnter message: ").strip()

        if user_message.lower() == "exit":
            print("Goodbye!")
            break

        if not user_message:
            print("Please enter a message.")
            continue

        result, confidence = predict_message(spam_model, user_message)
        print(f"Prediction: {result.upper()}")
        print(f"Confidence: {confidence:.2%}")
