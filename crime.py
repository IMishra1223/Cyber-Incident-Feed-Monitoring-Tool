import os
from pathlib import Path
import sys
import argparse
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Output directory for artifacts
OUT_DIR = Path("cyber_ml_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def try_load_csv(path="cyber_crimes.csv"):
    """
    Try to load CSV with several separators. Returns pandas.DataFrame.
    Adjusted to prioritize comma and robustly handle single-column reads.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"'{path}' not found. Place the dataset in the working directory or provide a path.")

    separators_to_try = [",", "\t", ";"] # Prioritize comma based on prior inspection
    for sep in separators_to_try:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
            # If a single column is loaded but the content of the first cell looks like CSV, try another separator.
            # This handles cases where a non-comma separator results in a single column that actually contains CSV data.
            if df.shape[1] == 1 and sep != "," and df.iloc[0, 0].count(',') > 2: # heuristic: more than 2 commas means likely CSV
                print(f"[INFO] Loaded with sep='{sep}' resulted in single column. Re-trying with other separators.")
                continue # Skip this result and try next separator
            print(f"[INFO] Loaded '{path}' with sep='{sep}', shape={df.shape}")
            return df
        except Exception as e:
            # print(f"Debug: Failed with sep='{sep}': {e}") # For debugging if needed
            continue
    # if we reach here, try pandas autodetect (less reliable)
    try:
        df = pd.read_csv(path, encoding="utf-8", sep=None, engine='python') # Let pandas auto-detect
        print(f"[INFO] Loaded '{path}' with default separator (auto-detected), shape={df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Could not read '{path}': {e}")

def detect_columns(df):
    """
    Heuristic detection of text (Description) and label (Category) columns.
    Returns (text_col, label_col)
    """
    possible_text = ["Description","description","Details","details","Crime Description","Crime_Description","Remarks","remarks","Narrative"]
    possible_label = ["Category","category","Type","type","Crime_Type","CrimeType","Label"]
    text_col = None
    label_col = None
    for c in possible_text:
        if c in df.columns:
            text_col = c
            break
    for c in possible_label:
        if c in df.columns:
            label_col = c
            break
    return text_col, label_col

def prepare_features(df, text_col=None, label_col=None, max_tfidf_features=2000):
    """
    Prepare features and labels for ML.
    - If label_col missing but year columns present -> create binary label from median of totals.
    - If text_col present -> TF-IDF vectorize text and optionally one-hot encode 'State/UT' column.
    - Else -> use numeric year columns and State one-hot if available.
    Returns: X (dense or sparse), y_enc (numpy array of encoded labels), label_encoder, feature_description, additional artifacts (tfidf, ohe)
    """
    df_proc = df.copy()
    # detect year columns like '2016' or '2017'
    year_cols = [c for c in df_proc.columns if c.isdigit() or c.startswith("201")]
    if label_col is None and year_cols:
        df_proc["total_cases"] = df_proc[year_cols].sum(axis=1)
        median = df_proc["total_cases"].median()
        df_proc["Label"] = np.where(df_proc["total_cases"] > median, "High", "Low")
        label_col = "Label"
        print("[INFO] No label column found. Created binary 'Label' from totals (High/Low).")
    if label_col is None:
        # final fallback: try to use a small set of columns for label-like info (not recommended)
        raise ValueError("No label column found and unable to create one. Please provide a 'Category' or 'Type' column.")
    y = df_proc[label_col].astype(str).values
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    # Feature creation
    tfidf = None
    ohe = None
    feature_desc = ""
    if text_col and text_col in df_proc.columns:
        tfidf = TfidfVectorizer(max_features=max_tfidf_features, stop_words="english")
        X_text = tfidf.fit_transform(df_proc[text_col].astype(str))
        feature_desc = f"TF-IDF({X_text.shape[1]} features)"
        # if State/UT exists, one-hot encode and horizontally stack
        if "State/UT" in df_proc.columns:
            ohe = OneHotEncoder(handle_unknown="ignore") # Removed sparse=False
            state_ohe = ohe.fit_transform(df_proc[["State/UT"]].astype(str)).toarray() # Convert to dense
            try:
                from scipy.sparse import hstack
                X = hstack([X_text, state_ohe])
                feature_desc += " + State one-hot"
            except Exception:
                X = np.hstack([X_text.toarray(), state_ohe])
                feature_desc += " + State one-hot (dense fallback)"
        else:
            X = X_text
    else:
        # No text: use numeric year columns if available
        if year_cols:
            X = df_proc[year_cols].fillna(0).values
            feature_desc = "Numeric year columns"
            if "State/UT" in df_proc.columns:
                ohe = OneHotEncoder(handle_unknown="ignore") # Removed sparse=False
                state_ohe = ohe.fit_transform(df_proc[["State/UT"]].astype(str)).toarray() # Convert to dense
                X = np.hstack([X, state_ohe])
                feature_desc += " + State one-hot"
        else:
            X = np.arange(len(df_proc)).reshape(-1,1)
            feature_desc = "Index fallback feature (no text/no year columns)"
    return X, y_enc, label_encoder, feature_desc, tfidf, ohe

def train_models(X, y_enc, label_encoder, test_size=0.2, random_state=42):
    """
    Train three classifiers and evaluate them.
    Returns a results dict with accuracy, report, and confusion matrix for each model, and the trained models.
    Saves confusion matrices and a CSV summary in OUT_DIR.
    """
    # determine stratify suitability
    unique, counts = np.unique(y_enc, return_counts=True)
    strat = y_enc if counts.min() >= 2 else None
    if strat is None:
        print("[WARN] Some classes have <2 samples. Not using stratify for train_test_split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state, stratify=strat)
    # choose models appropriate for sparse/dense
    use_text = hasattr(X_train, "toarray") or (len(X_train.shape)==2 and (not np.issubdtype(X_train.dtype, np.number)))
    models = {}
    if use_text:
        models["LinearSVC"] = LinearSVC(max_iter=5000)
        models["LogisticRegression"] = LogisticRegression(max_iter=5000)
        models["MultinomialNB"] = MultinomialNB()
    else:
        models["LinearSVC"] = LinearSVC(max_iter=5000)
        models["LogisticRegression"] = LogisticRegression(max_iter=5000)
        models["GaussianNB"] = GaussianNB()
    results = {}
    trained_classifiers = {}
    for name, clf in models.items():
        try:
            # Convert to dense arrays as necessary (GaussianNB needs dense)
            Xtr = X_train.toarray() if hasattr(X_train, "toarray") and name == "GaussianNB" else X_train
            Xte = X_test.toarray() if hasattr(X_test, "toarray") and name == "GaussianNB" else X_test
            # Some classifiers accept sparse, others do not; scikit handles many cases
...             clf.fit(Xtr, y_train)
...             y_pred = clf.predict(Xte)
...             acc = accuracy_score(y_test, y_pred)
... 
...             # Get unique numerical labels present in the current test set and predictions
...             actual_labels_in_report = np.unique(np.concatenate((y_test, y_pred)))
... 
...             # Pass these actual numerical labels to the 'labels' parameter of classification_report
...             # This ensures the report is generated only for classes present in the split.
...             rep = classification_report(y_test, y_pred,
...                                         labels=actual_labels_in_report,
...                                         target_names=list(label_encoder.classes_),
...                                         zero_division=0)
...             cm = confusion_matrix(y_test, y_pred)
...             results[name] = {"accuracy": float(acc), "report": rep, "confusion_matrix": cm.tolist()}
...             trained_classifiers[name] = clf # Store the trained classifier
...             # save confusion matrix image
...             try:
...                 plt.figure(figsize=(5,4))
...                 plt.imshow(cm, interpolation="nearest")
...                 plt.title(f"Confusion Matrix - {name}")
...                 plt.ylabel("True label")
...                 plt.xlabel("Predicted label")
...                 plt.colorbar()
...                 for i in range(cm.shape[0]):
...                     for j in range(cm.shape[1]):
...                         plt.text(j, i, str(cm[i,j]), ha="center", va="center")
...                 fig_path = OUT_DIR / f"confusion_{name}.png"
...                 plt.tight_layout()
...                 plt.savefig(fig_path)
...                 plt.close()
...             except Exception as e:
...                 print(f"[WARN] Could not save confusion matrix image for {name}: {e}")
        except Exception as e:
            tb = traceback.format_exc()
            results[name] = {"error": str(e), "traceback": tb}
            print(f"[ERROR] Training failed for {name}: {e}")
    # Save CSV summary
    rows = []
    for k,v in results.items():
        rows.append({"Model": k, "Accuracy": v.get("accuracy", None)})
    try:
        pd.DataFrame(rows).to_csv(OUT_DIR / "model_summary.csv", index=False)
    except Exception as e:
        print(f"[WARN] Could not save model_summary.csv: {e}")
    return results, trained_classifiers

def predict_single(text, df_sample, tfidf, ohe, label_encoder, model):
    """
    Given a single text (description) and the transformers, produce a predicted label.
    df_sample: DataFrame row(s) used to infer State/UT presence and columns.
    """
    if tfidf is None:
        raise ValueError("TF-IDF vectorizer not available. Prediction on text requires TF-IDF fitted on training data.")
    X_text = tfidf.transform([text])
    # if State/UT exists and ohe available, try to append a zero-state vector
    if ohe is not None and "State/UT" in df_sample.columns:
        # Default state vector -> zeros (unknown state)
        state_vec = ohe.transform(df_sample[["State/UT"]].astype(str)).toarray() # Convert to dense
        try:
            from scipy.sparse import hstack
            X = hstack([X_text, state_vec])
        except Exception:
            X = np.hstack([X_text.toarray(), state_vec])
    else:
        X = X_text

    # Ensure X is dense if the model expects it (e.g., GaussianNB)
    if hasattr(model, 'predict_proba') and isinstance(model, GaussianNB) or not hasattr(X, 'toarray'):
        X_pred = X.toarray() if hasattr(X, 'toarray') else X
    else:
        X_pred = X

    prediction_encoded = model.predict(X_pred)
    predicted_label = label_encoder.inverse_transform(prediction_encoded)
    return predicted_label[0]

def save_report_text(results, feature_desc, label_encoder):
    """
    Save a simple text report summarizing results and feature strategy.
    """
    lines = []
    lines.append("Cyber Crime ML Project - Summary Report")
    lines.append("=======================================")
    lines.append(f"Feature strategy: {feature_desc}")
    lines.append("Models trained:")
    for k,v in results.items():
        lines.append(f"- {k}: Accuracy = {v.get('accuracy', 'N/A')}")
    # save
    with open(OUT_DIR / "Cyber_Crime_ML_Project_Report.txt", "w") as f:
        f.write("\n".join(lines))
    print("[INFO] Summary report saved to", OUT_DIR / "Cyber_Crime_ML_Project_Report.txt")

def interactive_menu(original_df, X=None, y_enc=None, label_encoder=None, tfidf=None, ohe=None):
    """
    Interactive CLI: includes original viewing/search functions plus ML training & predict
    """
    df = original_df.copy()
    # These variables need to be updated by run_ml_pipeline for predict_now to potentially work
    current_tfidf = tfidf
    current_ohe = ohe
    current_label_encoder = label_encoder
    current_models = {} # New variable to store trained models

    def view_state_data():
        state = input("Enter State/UT name: ").strip().lower()
        res = df[df["State/UT"].astype(str).str.lower().str.contains(state)]
        if res.empty:
            print("No matching state found.")
        else:
            print(res.to_string(index=False))

    def search_keyword():
        key = input("Enter keyword to search (across all columns): ").strip().lower()
        res = df[df.astype(str).apply(lambda x: x.str.lower().str.contains(key)).any(axis=1)]
        if res.empty:
            print("No matching records found.")
        else:
            print(res.to_string(index=False))

    def yearly_summary():
        print("Yearly totals (detected year columns):")
        years = [c for c in df.columns if c.isdigit() or c.startswith("201")]
        if not years:
            print("No year columns detected.")
            return
        for y in years:
            print(f"Total in {y}: {df[y].sum()}")

    def show_full_dataset():
        print(df.to_string(index=False))

    def run_ml_pipeline():
        nonlocal current_tfidf, current_ohe, current_label_encoder, current_models # Declare intent to modify
        print("\n--- Running ML Pipeline (train & evaluate) ---")
        try:
            Xp, ype, le, feat_desc, tf, oh = prepare_features(df, *detect_columns(df))
            results, trained_clfs = train_models(Xp, ype, le) # Capture trained classifiers
            save_report_text(results, feat_desc, le)
            print("ML pipeline complete. Check cyber_ml_output/ for artifacts.")
            # Store transformers and models for prediction later in predict_now
            current_tfidf = tf
            current_ohe = oh
            current_label_encoder = le
            current_models = trained_clfs
        except Exception as e:
            print("ML pipeline failed:", e)
            traceback.print_exc()

    def predict_now():
        text = input("Enter a short description to predict: ").strip()
        if current_tfidf is None or current_label_encoder is None or not current_models:
            print("Predictor not fully available (TF-IDF, LabelEncoder, or models not fitted from ML pipeline). Run ML pipeline first.")
            return

        # Use one of the trained models for prediction, e.g., LinearSVC
        # You can change 'LinearSVC' to 'LogisticRegression' or 'MultinomialNB'/'GaussianNB' if desired.
        model_to_use = current_models.get("LinearSVC")
        if model_to_use is None:
            print("LinearSVC model not found. Please ensure it was trained successfully.")
            return

        try:
            predicted_label = predict_single(text, df.head(1), current_tfidf, current_ohe, current_label_encoder, model_to_use)
            print(f"\nPredicted Category for '{text}': {predicted_label}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()

    menu = {
        "1": ("View State-wise Cyber Crime Data", view_state_data),
        "2": ("Search by Keyword", search_keyword),
        "3": ("View Yearly Crime Summary", yearly_summary),
        "4": ("Show Entire Dataset", show_full_dataset),
        "5": ("Run ML Pipeline (train & evaluate)", run_ml_pipeline),
        "6": ("Prediction (single description) - helper", predict_now),
        "7": ("Exit", lambda: sys.exit(0))
    }

    # Automatically run ML Pipeline (option 5) and then Prediction (option 6)
    print("\n===== AUTOMATED CYBER CRIME MONITORING TOOL (ML) WORKFLOW ====")
    run_ml_pipeline()
    predict_now()
    print("\nAutomated workflow complete. Exiting.")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Cyber Crime ML - Clean final script (Option A)")
    parser.add_argument("--data", type=str, default="cyber_crimes.csv", help="Path to cyber_crimes.csv")
    parser.add_argument("--no-interactive", action="store_true", help="Run pipeline non-interactively and exit")
    args = parser.parse_args(argv)
    try:
        df = try_load_csv(args.data)
    except Exception as e:
        print("Failed to load dataset:", e)
        return
    text_col, label_col = detect_columns(df)
    print("[INFO] Detected text column:", text_col, "label column:", label_col)
    # If non-interactive, run pipeline and exit
    if args.no_interactive:
        try:
            X, y_enc, le, feat_desc, tfidf, ohe = prepare_features(df, text_col, label_col)
            results, _ = train_models(X, y_enc, le)
            save_report_text(results, feat_desc, le)
            print("[INFO] Pipeline finished. Outputs in", OUT_DIR)
        except Exception as e:
            print("Pipeline failed:", e)
            traceback.print_exc()
        return
    # Else launch interactive menu (also provides ML pipeline option)
    
    # First, run the ML pipeline and a sample prediction automatically
    print("\n===== Initializing ML Pipeline and Sample Prediction ====")
    temp_tfidf = None
    temp_ohe = None
    temp_le = None
    temp_models = {}
    try:
        Xp, ype, le, feat_desc, tf, oh = prepare_features(df, *detect_columns(df))
        results, trained_clfs = train_models(Xp, ype, le)
        save_report_text(results, feat_desc, le)
        temp_tfidf = tf
        temp_ohe = oh
        temp_le = le
        temp_models = trained_clfs
        print("ML Pipeline initialized. Ready for interactive use.")
    except Exception as e:
        print("Initial ML pipeline failed:", e)
        traceback.print_exc()

    # Then, launch the interactive menu with the initialized components
    interactive_menu_interactive(df, temp_tfidf, temp_ohe, temp_le, temp_models)


def interactive_menu_interactive(original_df, tfidf, ohe, label_encoder, trained_models):
    df = original_df.copy()

    def view_state_data_inter():
        state = input("Enter State/UT name: ").strip().lower()
        res = df[df["State/UT"].astype(str).str.lower().str.contains(state)]
        if res.empty:
            print("No matching state found.")
        else:
            print(res.to_string(index=False))

    def search_keyword_inter():
        key = input("Enter keyword to search (across all columns): ").strip().lower()
        res = df[df.astype(str).apply(lambda x: x.str.lower().str.contains(key)).any(axis=1)]
        if res.empty:
            print("No matching records found.")
        else:
            print(res.to_string(index=False))

    def yearly_summary_inter():
        print("Yearly totals (detected year columns):")
        years = [c for c in df.columns if c.isdigit() or c.startswith("201")]
        if not years:
            print("No year columns detected.")
            return
        for y in years:
            print(f"Total in {y}: {df[y].sum()}")

    def show_full_dataset_inter():
        print(df.to_string(index=False))

    def run_ml_pipeline_inter():
        print("\nML pipeline has already been initialized. Rerunning may overwrite current models.")
        choice = input("Do you want to rerun the ML pipeline? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                Xp, ype, le, feat_desc, tf, oh = prepare_features(df, *detect_columns(df))
                results, trained_clfs = train_models(Xp, ype, le)
                save_report_text(results, feat_desc, le)
                print("ML pipeline complete. Check cyber_ml_output/ for artifacts.")
                # Update the main interactive_menu's variables if re-run
                nonlocal tfidf, ohe, label_encoder, trained_models
                tfidf = tf
                ohe = oh
                label_encoder = le
                trained_models = trained_clfs
            except Exception as e:
                print("ML pipeline failed:", e)
                traceback.print_exc()
        else:
            print("ML pipeline rerun cancelled.")

    def predict_now_inter():
        text = input("Enter a short description to predict: ").strip()
        if tfidf is None or label_encoder is None or not trained_models:
            print("Predictor not fully available. Run ML pipeline first (Option 5).")
            return

        model_to_use = trained_models.get("LinearSVC")
        if model_to_use is None:
            print("LinearSVC model not found. Please ensure it was trained successfully.")
            return

        try:
            predicted_label = predict_single(text, df.head(1), tfidf, ohe, label_encoder, model_to_use)
            print(f"\nPredicted Category for '{text}': {predicted_label}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()

    menu = {
        "1": ("View State-wise Cyber Crime Data", view_state_data_inter),
        "2": ("Search by Keyword", search_keyword_inter),
        "3": ("View Yearly Crime Summary", yearly_summary_inter),
        "4": ("Show Entire Dataset", show_full_dataset_inter),
        "5": ("Run ML Pipeline (train & evaluate)", run_ml_pipeline_inter),
        "6": ("Prediction (single description)", predict_now_inter),
        "7": ("Exit", lambda: sys.exit(0))
    }

    while True:
        print("\n===== CYBER CRIME MONITORING TOOL (ML) ====")
        for k,v in menu.items():
            print(f"{k}. {v[0]}")
        choice = input("Enter your choice: ").strip()
        if choice in menu:
            try:
                menu[choice][1]()
            except SystemExit:
                raise
            except Exception as e:
                print("Error during operation:", e)
                traceback.print_exc()
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main(argv=[]) # Pass an empty list to argv to prevent argparse from processing kernel arguments
