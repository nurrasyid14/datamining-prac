import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.prep.normalisation import Normalisation
from src.classification.knn import KNN

# Data
train_df = pd.read_csv('data/titanic.csv')
test_df = pd.read_csv('data/titanic/test.csv')
test_label_df = pd.read_csv('data/titanic/testlabel.csv')

target_cols = ['Age', 'Fare']

# --- Train ---
pos_missing_train = train_df[target_cols].isnull().any(axis=1)

train_data = train_df.loc[~pos_missing_train, target_cols].reset_index(drop=True)
train_label = train_df.loc[~pos_missing_train, 'Survived'].reset_index(drop=True)

# --- Test ---
test_combined = test_df.merge(
    test_label_df,
    on="PassengerId"
)

pos_missing_test = test_combined[target_cols].isnull().any(axis=1)

test_clean = test_combined.loc[~pos_missing_test].reset_index(drop=True)

test_data = test_clean[target_cols]
test_label = test_clean['Survived']

# --- Scaling ---
norm = Normalisation(method="minmax", use_scikit=False)
train_data_norm = norm.fit_transform(train_data)
test_data_norm = norm.transform(test_data)

# --- KNN ---
error_ratios = {}

for k in range(1, 16):
    model = KNN(data=train_data_norm, target=train_label, k=k)

    model.fit()
    preds = model.predict(test_data_norm)

    errors = (preds != test_label).sum()
    total = len(test_label)

    error_ratio = errors / total
    error_ratios[k] = error_ratio

    print(f"k={k}, error ratio={error_ratio:.4f}")

# --- Kesimpulan ---
print("\nFinal Error Ratios:")
for k, v in error_ratios.items():
    print(f"k={k}: {v:.4f}")