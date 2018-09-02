import pandas as pd
import sys
import os
# first_file = sys.argv[1]
# second_file = sys.argv[2]

def corr(first_file, second_file):
  first_df = pd.read_csv(first_file,index_col=0)
  second_df = pd.read_csv(second_file,index_col=0)
  # assuming first column is `prediction_id` and second column is `prediction`
  prediction = first_df.columns[0]
  # correlation
  print("Finding correlation between: {} and {}".format(first_file,second_file))
  print("Column to be measured: {}".format(prediction))
  print("Pearson's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='pearson')))
  print("Kendall's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='kendall')))
  print("Spearman's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='spearman')))

local_path='../files/'
for root, dirs, files in os.walk(local_path):
  for file1 in files:
    for file2 in files:
      file_path1 = os.path.join(root, file1)
      file_path2 = os.path.join(root, file2)
      corr(file_path1, file_path2)
