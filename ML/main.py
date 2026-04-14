from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dataset import preprocess_data
from model import SVMModel


RESULTS_DIR = Path(__file__).resolve().parent / "results"
LABELS = ["A", "C"]
TARGET_NAMES = ["AD", "Control"]


def save_confusion_matrix_heatmap(confusion_mat, output_path):
    """
    Saves the confusion matrix as a heatmap image.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_metrics_bar_chart(report_dict, output_path):
    """
    Saves a bar chart for precision, recall, and F1 by class.
    """
    classes = TARGET_NAMES

    precision_vals = [report_dict[class_name]["precision"] for class_name in classes]
    recall_vals = [report_dict[class_name]["recall"] for class_name in classes]
    f1_vals = [report_dict[class_name]["f1-score"] for class_name in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision_vals, width, label="Precision")
    plt.bar(x, recall_vals, width, label="Recall")
    plt.bar(x + width, f1_vals, width, label="F1 Score")

    plt.xticks(x, classes)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load processed dataset
    preprocessed_df = preprocess_data()

    # Features (EEG bands)
    X = preprocessed_df[["Delta", "Theta", "Alpha", "Beta"]]

    # Labels (A = Alzheimer's, C = Control)
    y = preprocessed_df["Group"]

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_ad_sensitivities = []

    # Store all predictions across folds for final plots
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Print shapes and class counts
        print(f"\n=== Fold {fold} ===")
        print("Full dataset shape:", preprocessed_df.shape)
        print("Training shape:", X_train_scaled.shape)
        print("Testing shape:", X_test_scaled.shape)

        print("\nTrain groups:")
        print(y_train.value_counts())

        print("\nTest groups:")
        print(y_test.value_counts())

        # Initialize and train base linear SVM model
        model = SVMModel()
        model.train(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Person C: full evaluation
        accuracy = np.mean(y_pred == y_test)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        report = classification_report(
            y_test,
            y_pred,
            labels=LABELS,
            target_names=TARGET_NAMES,
            zero_division=0,
        )
        ad_sensitivity = recall_score(y_test, y_pred, pos_label="A", zero_division=0)

        print("\nAccuracy:", round(accuracy, 4))
        print("Confusion Matrix (rows=true, cols=pred) [A, C]:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        print("AD Sensitivity (Recall for class 'A'):", round(ad_sensitivity, 4))

        fold_accuracies.append(accuracy)
        fold_ad_sensitivities.append(ad_sensitivity)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    overall_cm = confusion_matrix(all_y_true, all_y_pred, labels=LABELS)
    overall_report_text = classification_report(
        all_y_true,
        all_y_pred,
        labels=LABELS,
        target_names=TARGET_NAMES,
        zero_division=0,
    )
    overall_report_dict = classification_report(
        all_y_true,
        all_y_pred,
        labels=LABELS,
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )

    cm_plot_path = RESULTS_DIR / "confusion_matrix_heatmap.png"
    metrics_plot_path = RESULTS_DIR / "classification_metrics_bar_chart.png"

    save_confusion_matrix_heatmap(overall_cm, cm_plot_path)
    save_metrics_bar_chart(overall_report_dict, metrics_plot_path)

    print("\nFinal Results")
    print("Fold Accuracies:", fold_accuracies)
    print("Mean Accuracy:", np.mean(fold_accuracies))
    print("Fold AD Sensitivities:", fold_ad_sensitivities)
    print("Mean AD Sensitivity:", np.mean(fold_ad_sensitivities))

    print("\nOverall Confusion Matrix (all CV predictions):")
    print(overall_cm)

    print("\nOverall Classification Report:")
    print(overall_report_text)

    print("\nSaved plots:")
    print(cm_plot_path)
    print(metrics_plot_path)


if __name__ == "__main__":
    main()
