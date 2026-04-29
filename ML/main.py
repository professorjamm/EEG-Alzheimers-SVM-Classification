from pathlib import Path

import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler

from dataset import preprocess_data
from model import SVMModel


RESULTS_DIR = Path(__file__).resolve().parent / "results"
LABELS = ["A", "C"]
TARGET_NAMES = ["AD", "Control"]


def run_feature_subset_experiments(subject_df, output_dir):
    """
    Week 3 Person C:
    Compare feature subsets (Alpha only, Alpha+Delta, All Bands) with CV.
    """
    feature_subsets = {
        "alpha_only": ["Alpha"],
        "alpha_delta": ["Alpha", "Delta"],
        "all_bands": ["Delta", "Theta", "Alpha", "Beta"],
    }

    output_dir.mkdir(exist_ok=True)
    summary_rows = []

    for subset_name, feature_columns in feature_subsets.items():
        subset_dir = output_dir / subset_name
        subset_dir.mkdir(exist_ok=True)

        X = subject_df[feature_columns]
        y = subject_df["Group"]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_ad_sensitivities = []
        all_y_true = []
        all_y_pred = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVMModel(kernel="linear")
            model.train(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            accuracy = np.mean(y_pred == y_test)
            ad_sensitivity = recall_score(y_test, y_pred, pos_label="A", zero_division=0)

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

        cm_plot_path = subset_dir / "confusion_matrix_heatmap.png"
        metrics_plot_path = subset_dir / "classification_metrics_bar_chart.png"
        classification_report_path = subset_dir / "classification_report.txt"
        params_path = subset_dir / "model_params.txt"

        save_confusion_matrix_heatmap(overall_cm, cm_plot_path)
        save_metrics_bar_chart(overall_report_dict, metrics_plot_path)
        with open(classification_report_path, "w") as f:
            f.write(overall_report_text)
        with open(params_path, "w") as f:
            f.write("Kernel: linear\n")
            f.write(f"Features: {', '.join(feature_columns)}\n")

        mean_accuracy = float(np.mean(fold_accuracies))
        mean_ad_sensitivity = float(np.mean(fold_ad_sensitivities))

        summary_rows.append(
            {
                "subset": subset_name,
                "features": ", ".join(feature_columns),
                "mean_accuracy": round(mean_accuracy, 4),
                "mean_ad_sensitivity": round(mean_ad_sensitivity, 4),
                "ad_precision": round(overall_report_dict["AD"]["precision"], 4),
                "ad_recall": round(overall_report_dict["AD"]["recall"], 4),
                "ad_f1": round(overall_report_dict["AD"]["f1-score"], 4),
                "control_precision": round(overall_report_dict["Control"]["precision"], 4),
                "control_recall": round(overall_report_dict["Control"]["recall"], 4),
                "control_f1": round(overall_report_dict["Control"]["f1-score"], 4),
            }
        )

        print(f"\n[Person C] Feature subset: {subset_name}")
        print("Features:", feature_columns)
        print("Mean Accuracy:", round(mean_accuracy, 4))
        print("Mean AD Sensitivity:", round(mean_ad_sensitivity, 4))
        print("Saved:", cm_plot_path)
        print("Saved:", metrics_plot_path)
        print("Saved:", classification_report_path)

    summary_rows.sort(key=lambda row: row["mean_accuracy"], reverse=True)
    summary_path = output_dir / "feature_subset_comparison.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n[Person C] Feature subset comparison saved:", summary_path)
    print("[Person C] Best subset by mean accuracy:", summary_rows[0]["subset"])
    return summary_rows


def run_hyperparameter_tuning(X, y, output_dir):
    results_path = output_dir / "grid_search_results.csv"

    param_grid = [
        {
            "kernel": ["linear"],
            "C": [0.01, 0.1, 1, 10, 100],
        },
        {
            "kernel": ["rbf"],
            "C": [0.01, 0.1, 1, 10, 100],
            "gamma": ["scale", 0.001, 0.01, 0.1, 1],
        },
        {
            "kernel": ["poly"],
            "C": [0.01, 0.1, 1, 10, 100],
            "degree": [2, 3, 4],
        },
    ]

    best_config = {
        "config_num": 0,
        "accuracy": 0.0,
        "params": None
    }

    rows = []

    for i, params in enumerate(ParameterGrid(param_grid)):
        kernel = params["kernel"]
        C = params["C"]
        gamma = params.get("gamma", "scale")
        degree = params.get("degree", 3)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_accuracies = []
        fold_train_accuracies = []
        fold_ad_sensitivities = []

        # Store all predictions across folds for final plots
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"Beginning fold {fold}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize and train SVM model
            model = SVMModel(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree
            )
            model.train(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            train_pred = model.predict(X_train_scaled)

            accuracy = np.mean(y_pred == y_test)
            train_acc = np.mean(train_pred == y_train)

            report = classification_report(
                y_test,
                y_pred,
                labels=LABELS,
                target_names=TARGET_NAMES,
                zero_division=0,
            )
            ad_sensitivity = recall_score(y_test, y_pred, pos_label="A", zero_division=0)

            print(f"\nFold {fold} Accuracy:", round(accuracy, 4))
            print(f"\nFold {fold} AD Sensitivity (Recall for class 'A'):", round(ad_sensitivity, 4))
            print(f"\nFold {fold} Confusion Matrix (rows=true, cols=pred) [A, C]:")
            print(confusion_matrix(y_test, y_pred, labels=LABELS))
            print(f"\nFold {fold} Classification Report:")
            print(report)

            fold_accuracies.append(accuracy)
            fold_train_accuracies.append(train_acc)
            fold_ad_sensitivities.append(ad_sensitivity)

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
        
        # Make configuration results folder
        (output_dir  / f"config_{i}").mkdir(exist_ok=True)
        
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

        cm_plot_path = output_dir / f"config_{i}" / "confusion_matrix_heatmap.png"
        metrics_plot_path = output_dir  / f"config_{i}" / "classification_metrics_bar_chart.png"
        classification_report_path = output_dir  / f"config_{i}" / "classification_report.txt"
        params_path = output_dir  / f"config_{i}" / "model_params.txt"

        mean_test_score = float(np.mean(fold_accuracies))
        std_test_score = float(np.std(fold_accuracies))
        mean_train_score = float(np.mean(fold_train_accuracies))
        std_train_score = float(np.std(fold_train_accuracies))

        rows.append({
            "config_num": i,
            "kernel": kernel,
            "mean_test_score": round(mean_test_score, 4),
            "std_test_score": round(std_test_score, 4),
            "mean_train_score": round(mean_train_score, 4),
            "std_train_score": round(std_train_score, 4),
            "mean_ad_sensitivity": round(np.mean(fold_ad_sensitivities), 4),
            "std_ad_sensitivity": round(np.std(fold_ad_sensitivities), 4),
            "params": params,
        })

        save_confusion_matrix_heatmap(overall_cm, cm_plot_path)
        save_metrics_bar_chart(overall_report_dict, metrics_plot_path)
        with open(classification_report_path, "w") as f:
            f.write(overall_report_text)
        with open(params_path, "w") as f:
            f.write(str(params))

        print("\nFinal Results")
        print("Fold Accuracies:", fold_accuracies)
        print("Mean Accuracy:", mean_test_score)
        print("Fold AD Sensitivities:", fold_ad_sensitivities)
        print("Mean AD Sensitivity:", np.mean(fold_ad_sensitivities))

        print("\nOverall Confusion Matrix (all CV predictions):")
        print(overall_cm)

        print("\nOverall Classification Report:")
        print(overall_report_text)

        # Update best config based on mean accuracy
        if mean_test_score > best_config["accuracy"]:
            best_config["accuracy"] = mean_test_score
            best_config["config_num"] = i
            best_config["params"] = params

    rows.sort(key=lambda row: row["mean_test_score"], reverse=True)

    # Assign ranks manually
    for rank, row in enumerate(rows, start=1):
        row["rank_test_score"] = rank

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_num",
                "rank_test_score",
                "kernel",
                "mean_test_score",
                "std_test_score",
                "mean_train_score",
                "std_train_score",
                "mean_ad_sensitivity",
                "std_ad_sensitivity",
                "params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return best_config, rows


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


def save_kernel_comparison_chart(kernel_rows, output_path, title):
    """
    Saves a chart comparing kernel performance.
    """
    kernels = [row["kernel"].upper() for row in kernel_rows]
    accuracies = [row["mean_test_score"] for row in kernel_rows]
    sensitivities = [row["mean_ad_sensitivity"] for row in kernel_rows]

    x = np.arange(len(kernels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, accuracies, width, label="Mean Accuracy")
    plt.bar(x + width / 2, sensitivities, width, label="Mean AD Sensitivity")
    plt.xticks(x, kernels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_subset_comparison_chart(summary_rows, output_path):
    """
    Saves a chart comparing feature subset performance.
    """
    subset_labels = [row["subset"].replace("_", "\n") for row in summary_rows]
    accuracies = [row["mean_accuracy"] for row in summary_rows]
    sensitivities = [row["mean_ad_sensitivity"] for row in summary_rows]

    x = np.arange(len(subset_labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, accuracies, width, label="Mean Accuracy")
    plt.bar(x + width / 2, sensitivities, width, label="Mean AD Sensitivity")
    plt.xticks(x, subset_labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Feature Subset Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_representation_comparison_chart(rep_rows, output_path):
    """
    Saves a chart comparing subject vs channel representations.
    """
    labels = [row["representation"].title() for row in rep_rows]
    accuracies = [row["accuracy"] for row in rep_rows]
    sensitivities = [row["ad_sensitivity"] for row in rep_rows]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, accuracies, width, label="Test Accuracy")
    plt.bar(x + width / 2, sensitivities, width, label="AD Sensitivity")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Feature Representation Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_best_row_per_kernel(rows):
    best_rows = {}

    for row in rows:
        kernel = row["kernel"]
        if kernel not in best_rows or row["mean_test_score"] > best_rows[kernel]["mean_test_score"]:
            best_rows[kernel] = row

    ordered_rows = []
    for kernel_name in ["linear", "rbf", "poly"]:
        if kernel_name in best_rows:
            ordered_rows.append(best_rows[kernel_name])

    return ordered_rows


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    final_visualizations_dir = RESULTS_DIR / "final_visualizations"
    final_visualizations_dir.mkdir(exist_ok=True)

    # Load processed dataset
    representations = {
        "subject": preprocess_data(mode="subject"),
        "channel": preprocess_data(mode="channel"),
    }

    # Week 3 Person C: feature subset experiment based on Task 3 findings.
    feature_subset_rows = run_feature_subset_experiments(
        subject_df=representations["subject"],
        output_dir=RESULTS_DIR / "feature_subsets",
    )

    save_feature_subset_comparison_chart(
        feature_subset_rows,
        final_visualizations_dir / "feature_subset_comparison_chart.png",
    )

    representation_rows = []
    best_final_model = {
        "accuracy": -1,
        "representation": None,
        "confusion_matrix": None,
        "report_dict": None,
    }

    for rep_name, preprocessed_df in representations.items():
        print(f"Running: {rep_name.upper()} FEATURES")

        rep_dir = RESULTS_DIR / rep_name
        rep_dir.mkdir(exist_ok=True)

        # Features (EEG bands)
        X = preprocessed_df.drop(columns = ["Subject", "Group"])

        # Labels (A = Alzheimer's, C = Control)
        y = preprocessed_df["Group"]

        # Create final test set for final model evaluation
        X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )

        best_config, tuning_rows = run_hyperparameter_tuning(X_train_full, y_train_full, rep_dir)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test_final)

        params = best_config["params"]

        model_kwargs = {
            "kernel": params["kernel"],
            "C": params["C"],
        }

        if "gamma" in params:
            model_kwargs["gamma"] = params["gamma"]

        if "degree" in params:
            model_kwargs["degree"] = params["degree"]

        final_model = SVMModel(**model_kwargs)
        final_model.train(X_train_scaled, y_train_full)

        y_pred = final_model.predict(X_test_scaled)

        print("\nFINAL TEST RESULTS")
        print(confusion_matrix(y_test_final, y_pred))
        print(classification_report(y_test_final, y_pred))

        cm = confusion_matrix(y_test_final, y_pred, labels=LABELS)
        overall_report_text = classification_report(
            y_test_final,
            y_pred,
            labels=LABELS,
            target_names=TARGET_NAMES,
            zero_division=0,
        )
        overall_report_dict = classification_report(
            y_test_final,
            y_pred,
            labels=LABELS,
            target_names=TARGET_NAMES,
            zero_division=0,
            output_dict=True,
        )

        cm_plot_path = rep_dir / "test_confusion_matrix_heatmap.png"
        metrics_plot_path = rep_dir / "test_classification_metrics_bar_chart.png"
        classification_report_path = rep_dir / "test_classification_report.txt"
        params_path = rep_dir / "model_params.txt"
        kernel_chart_path = rep_dir / "kernel_comparison_chart.png"

        ad_sensitivity = recall_score(y_test_final, y_pred, pos_label="A", zero_division=0)
        accuracy = accuracy_score(y_test_final, y_pred)

        best_kernel_rows = get_best_row_per_kernel(tuning_rows)
        save_kernel_comparison_chart(
            best_kernel_rows,
            kernel_chart_path,
            f"Kernel Comparison ({rep_name.title()} Features)",
        )
        save_kernel_comparison_chart(
            best_kernel_rows,
            final_visualizations_dir / f"{rep_name}_kernel_comparison_chart.png",
            f"Kernel Comparison ({rep_name.title()} Features)",
        )

        save_confusion_matrix_heatmap(cm, cm_plot_path)
        save_metrics_bar_chart(overall_report_dict, metrics_plot_path)
        with open(classification_report_path, "w") as f:
            f.write(overall_report_text)
            f.write(f"\nAD Sensitivity (Recall for class 'A'): {round(ad_sensitivity, 4)}\n")
        with open(params_path, "w") as f:
            f.write(str(params))

        print("\nFinal Results")
        print("Test Set Accuracy:", accuracy)

        print("\nTest Set AD Sensitivity:", ad_sensitivity)

        print("\nTest Set Confusion Matrix:")
        print(cm)

        print("\nOverall Classification Report:")
        print(overall_report_text)

        print("\nSaved files:")
        print(cm_plot_path)
        print(metrics_plot_path)
        print(classification_report_path)
        print(kernel_chart_path)

        representation_rows.append({
            "representation": rep_name,
            "accuracy": accuracy,
            "ad_sensitivity": ad_sensitivity,
        })

        if accuracy > best_final_model["accuracy"]:
            best_final_model["accuracy"] = accuracy
            best_final_model["representation"] = rep_name
            best_final_model["confusion_matrix"] = cm
            best_final_model["report_dict"] = overall_report_dict

    save_representation_comparison_chart(
        representation_rows,
        final_visualizations_dir / "representation_comparison_chart.png",
    )

    if best_final_model["confusion_matrix"] is not None:
        save_confusion_matrix_heatmap(
            best_final_model["confusion_matrix"],
            final_visualizations_dir / "best_final_model_confusion_matrix_heatmap.png",
        )
        save_metrics_bar_chart(
            best_final_model["report_dict"],
            final_visualizations_dir / "best_final_model_classification_metrics_bar_chart.png",
        )

        print("\nBest final model representation:", best_final_model["representation"])
        print("Best final model accuracy:", best_final_model["accuracy"])
        print("Saved:", final_visualizations_dir / "best_final_model_confusion_matrix_heatmap.png")
        print("Saved:", final_visualizations_dir / "best_final_model_classification_metrics_bar_chart.png")


if __name__ == "__main__":
    main()
