from src.prep.data_eng import Validator, Filler
from src.prep.normalisation import Normalisation
from src.metrics.classification_metrics import ClassificationMetrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

'''
Export the Splitted data into a csv file, with original file name as a folder, and the file name as "train.csv" and "test.csv"
'''

def export_splitted_data(X_train, X_test, y_train, y_test, original_file_name):
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(f"{original_file_name}/train.csv", index=False)
    test_df.to_csv(f"{original_file_name}/test.csv", index=False)

titanic = pd.read_csv("data/titanic.csv")

print(titanic.head())