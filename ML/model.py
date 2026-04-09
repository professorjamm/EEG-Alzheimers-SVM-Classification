from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


class SVMModel:
  def __init__(self, kernel='linear',C=1.0, gamma='scale', random_state=42):
    self.model = SVC(
      kernel=kernel,
      C=C,
      gamma=gamma,
      random_state=random_state
    )

  def train(self, X_train, y_train):
    self.model.fit(X_train, y_train)

  def predict(self, X_test):
    return self.model.predict(X_test)

  def evaluate(self, X_test, y_test):
      """
      NOTE: More metrics for evaluation will be added below
      """
      preds = self.predict(X_test)
      acc = accuracy_score(y_test, preds)
      report = classification_report(y_test, preds)

      return {
        "accuracy": acc,
        "report": report
      }
