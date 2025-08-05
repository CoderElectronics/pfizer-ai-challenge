import pandas as pd

gt_pts = pd.read_csv('groundtruth.csv')

gt_pts.drop('X POSITION', axis=1, inplace=True)
gt_pts.drop('Y POSITION', axis=1, inplace=True)
gt_pts.rename(columns={'TIMESTAMP': 'Ts'}, inplace=True)

gt_pts.to_csv('input.csv', index=False)
