PATH_TO_CUSTOMER_HASHES = '../input/amex-data-files/'
PROCESS_DATA = True
PATH_TO_DATA = './data/'
TRAIN_MODEL = True
PATH_TO_MODEL = './model/'
INFER_TEST = True

import pandas as pd

train_data = pd.read_feather('/home/vincent0110/test/Kaggle/data/train_data.ftr')
