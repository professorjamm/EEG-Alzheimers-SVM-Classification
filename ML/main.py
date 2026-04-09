from dataset import preprocess_data
from model import SVMModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Load processed dataset
    preprocessed_df = preprocess_data()

    # Features (EEG bands)
    X = preprocessed_df[["Delta", "Theta", "Alpha", "Beta"]]

    # Labels (A = Alzheimer's, C = Control)
    y = preprocessed_df["Group"]

    # Split data while keeping class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

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
    model.train(X_train, y_train)

    results = model.evaluate(X_test, y_test)
    print(results["report"])


if __name__ == "__main__":
    main()
