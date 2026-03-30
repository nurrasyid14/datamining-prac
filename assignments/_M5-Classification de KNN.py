from src.prep.data_eng import Validator, Filler
from src.prep.normalisation import Normalisation
from src.metrics.classification_metrics import ClassificationMetrics
from src.classification.knn import KNN
from sklearn.model_selection import train_test_split
import pandas as pd

'''
Export the Splitted data into a csv file, with original file name as a folder, and the file name as "train.csv" and "test.csv"
'''

#1 data calls

test = pd.read_csv('data/titanic/test.csv')