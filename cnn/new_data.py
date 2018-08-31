import csv
import pandas as pd
test = pd.read_csv('C:/work/input_data/test.csv')
new_ = pd.read_csv('valid_id')
# ge=test.join(new_, how ='inner',on='id')
geg=pd.merge(new_, test, how='inner', on=['id', 'id'])
train = pd.read_csv('C:/work/input_data/train.csv')
result = train.append(geg)
print('gg')