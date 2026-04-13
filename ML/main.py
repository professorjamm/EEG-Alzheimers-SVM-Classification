import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dataset import preprocess_data
from model import SVMModel


def main():
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
        labels = ["A", "C"]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report = classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=["AD", "Control"],
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

    print("\nFinal Results")
    print("Fold Accuracies:", fold_accuracies)
    print("Mean Accuracy:", np.mean(fold_accuracies))
    print("Fold AD Sensitivities:", fold_ad_sensitivities)
    print("Mean AD Sensitivity:", np.mean(fold_ad_sensitivities))


if __name__ == "__main__":
    main()
