import unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

class DatasetTests(unittest.TestCase):
    def test_original_data(self):
        df=pd.read_csv(ROOT/"data"/"CustomerChurn.csv")
        self.assertEqual(df.shape,(3150,14))
        self.assertEqual(df.columns[-1],"Churn")
        self.assertEqual(set(df.Churn.unique()),{0,1})
        self.assertEqual(int(df.isna().sum().sum()),0)

if __name__=="__main__":unittest.main()
