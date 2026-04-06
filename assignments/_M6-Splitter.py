import pathlib
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.prep.splitter import Splitter
from src.prep.data_eng import Validator, Filler
from src.prep.normalisation import Normalisation
from src.metrics.classification_metrics import ClassificationMetrics
from src.classification.knn import KNN


# Block 1 : Data Prep
data = pathlib.Path(__file__).parent.parent / 'data/titanic.csv'
data = pd.read_csv(data)
print(data.head())

# Block 2 : Data Engineering & Normalisation -- Titanic[Sex, Age, Pclass, Fare]

# fill age with mean of the classes
filler = Filler(data, method="mean")
filled_df = filler.fill()
age = filled_df['Age']
print(f"Filled Age with mean: {age.mean()}")

# Label : Survived
survived, died = data['Survived'].value_counts()
print(f"Survived: {survived}, Died: {died}")

# Block 3 : Data Splitting
splitter = Splitter(data, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1)
holdout_train, holdout_test, holdout_val = splitter.holdout(random_state=42)
kfolds = splitter.k_fold(random_state=42)
loo = splitter.LoO()

# Block 4 : Normalisation
normaliser = Normalisation(data)

# on train set
normalised_data = normaliser.min_max()
# on test set
normalised_test = normaliser.min_max(holdout_test)

# Block 5a : KNN Classification on train set
features = ['Sex', 'Age', 'Pclass', 'Fare']
X = normalised_data[features]
y = normalised_data['Survived']

knn = KNN(data=normalised_data, target='Survived', k=5)
knn.split()
knn.train()
predictions = knn.predict()
print(f"Predictions: {predictions[:10]}")
 
# Block 5b : KNN Classification on test set
X_test = normalised_test[features]
y_test = normalised_test['Survived']
knn_test = KNN(data=normalised_test, target='Survived', k=5)
knn_test.split()
knn_test.train()
test_predictions = knn_test.predict()
print(f"Test Predictions: {test_predictions[:10]}")

# Block 6 : Metrics
metrics = ClassificationMetrics(y_test, test_predictions)
accuracy = metrics.accuracy()
precision = metrics.precision()
recall = metrics.recall()
f1 = metrics.f1_score()
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")