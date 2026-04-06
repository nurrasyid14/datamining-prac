import pathlib
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prep.splitter import Splitter
from src.prep.data_eng import Validator, Filler
from src.prep.normalisation import Normalisation
from src.metrics.classification_metrics import ClassificationMetrics
from src.classification.knn import KNN

#===========================================================================

# Block 1 : Data Loading
path = pathlib.Path(__file__).parent.parent / "data" / "titanic.csv"
data = pd.read_csv(path)

print("=== Block 1: Raw Data Preview ===")
print(data.head(), "\n")

# Block 2 : Data Cleaning & Encoding
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

print("=== Block 2: After Encoding & Dropping Columns ===")
print(data.head(), "\n")

# Block 3 : Missing Value Handling
data = Filler(data, method="mean").fill()

print("=== Block 3: Missing Values Filled ===")
print("Remaining NaNs per column:")
print(data.isnull().sum(), "\n")

# Block 4 : Distribusi Label
survived, died = data['Survived'].value_counts()
print("=== Block 4: Label Distribution ===")
print(f"Survived: {survived}, Died: {died}\n")

# Block 5 : Data Splitting
splitter = Splitter(data, 0.7, 0.2, 0.1)
train, test, val = splitter.holdout(random_state=42)

print("=== Block 5: Data Split Sizes ===")
print(f"Train: {len(train)}, Test: {len(test)}, Val: {len(val)}\n")

# Block 6 : Normalisasi
normaliser = Normalisation(method="minmax")
normaliser.fit(train)

train = normaliser.transform(train)
test = normaliser.transform(test)

print("=== Block 6: Normalisation Applied ===")
print("Train sample:")
print(train.head(), "\n")

# Block 7 : Feature Selection
features = ['Sex', 'Age', 'Pclass', 'Fare']

X_train = train[features]
y_train = train['Survived']

X_test = test[features]
y_test = test['Survived']

print("=== Block 7: Features & Target ===")
print(f"Features: {features}")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\n")

# Block 8 : KNN Training
knn = KNN(k=5)
knn.fit(X_train, y_train)

print("=== Block 8: Model Training Completed ===\n")

# Block 9 : Prediction
preds = knn.predict(X_test)

print("=== Block 9: Predictions Sample ===")
print(preds[:10], "\n")

# Block 10 : Evaluation Metrics
metrics = ClassificationMetrics()

accuracy = metrics.accuracy(y_test, preds)
precision = metrics.precision(y_test, preds)
recall = metrics.recall(y_test, preds)
f1 = metrics.f1(y_test, preds)

print("=== Block 10: Evaluation Metrics ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")