from dataset import preprocess_data
from model import SVMModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

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

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Print shapes and class counts
        print("Full dataset shape:", preprocessed_df.shape)
        print("Training shape:", X_train_scaled.shape)
        print("Testing shape:", X_test_scaled.shape)

        print("\nTrain groups:")
        print(y_train.value_counts())

        print("\nTest groups:")
        print(y_test.value_counts())

    # Initialize the base linear SVM model
        model = SVMModel()
        model.train(X_train_scaled, y_train)

        results = model.evaluate(X_test, y_test)
        print("Accuracy:", results["accuracy"])
        print(results["report"])

        fold_accuracies.append(results["accuracy"])

    print("Final Results")
    print("Fold Accuracies:", fold_accuracies)
    print("Mean Accuracy:", np.mean(fold_accuracies))

if __name__ == "__main__":
    main()
