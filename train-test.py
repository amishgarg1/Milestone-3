import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("/content/preprocessed_fake_job_postings.csv")


X = df['final_text'].fillna("")
y = df['fraudulent']   


vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Model 1: Logistic Regression
# ---------------------------
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# ---------------------------
# Model 2: Random Forest
# ---------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    return acc


acc_log = evaluate_model("Logistic Regression", y_test, y_pred_log)
acc_rf = evaluate_model("Random Forest", y_test, y_pred_rf)


if acc_log >= acc_rf:
    best_model = log_model
    best_name = "Logistic Regression"
else:
    best_model = rf_model
    best_name = "Random Forest"

print(f"\n Best Model Selected: {best_name}")


joblib.dump(best_model, "/content/best_model.pkl")
joblib.dump(vectorizer, "/content/vectorizer.pkl")

print("\n Models saved as 'best_model.pkl' and 'vectorizer.pkl'")
