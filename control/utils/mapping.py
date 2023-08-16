import os
import pandas as pd
import numpy as np

class Mapping:
    def __init__(self, 
                 data_paths) -> None:
        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        self.data = []
        for data_path in data_paths:
            kind = os.path.basename(data_path).split('.')[1]
            if kind in  ['csv', 'txt']:
                self.data.append(pd.read_csv(data_path))
            elif kind == 'json':
                self.data.append(pd.read_json(data_path))
            else:
                raise TypeError("Unsupport data format.")
        new_df = self.data[0].query("Ticker == 'ACB'")
        new_df = new_df.dropna(axis=1, how='all')
        self.data[0] = self.data[0].dropna(axis=1, how='all')
        
        
        print(new_df)            
        # print(self.data[0].head(10))            
        
                
        pass


if __name__ == "__main__":
    file = f"/home/xuan/Project/OCR/label/stx_fsc_BalanceSheet.csv"
    m = Mapping(file)
