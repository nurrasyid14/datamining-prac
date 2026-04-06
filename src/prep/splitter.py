#splitter
import pandas as pd
import numpy as np

class Splitter:
    def __init__(
            self,
             data,
             train_ratio,
             test_ratio,
             val_ratio:float=0,
             fold:int=5
             ):
        self.data = data
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.fold = fold

    def holdout(self, random_state=None):
        shuffled_data = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        total = len(shuffled_data)

        if not np.isclose(self.train_ratio + self.test_ratio + self.val_ratio, 1.0):
            raise ValueError("Ratios must sum to 1")

        train_end = int(total * self.train_ratio)
        test_end = train_end + int(total * self.test_ratio)

        train = shuffled_data[:train_end]
        test = shuffled_data[train_end:test_end]
        val = shuffled_data[test_end:]  # absorb rounding remainder

        return train, test, val
    
    def k_fold(self, random_state=None):
        shuffled = self.data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        fold_sizes = np.full(self.fold, len(shuffled) // self.fold)
        fold_sizes[:len(shuffled) % self.fold] += 1  # distribute remainder

        folds = []
        current = 0
        for size in fold_sizes:
            folds.append(shuffled.iloc[current:current + size])
            current += size

        results = []
        for i in range(self.fold):
            val = folds[i]
            train = pd.concat(folds[:i] + folds[i+1:], ignore_index=True)
            results.append((train, val))

        return results
    
    def LoO(self):
        data = self.data.reset_index(drop=True)
        results = []
        for i in range(len(data)):
            test = data.iloc[[i]]
            train = data.drop(index=i)
            results.append((train, test))
        return results