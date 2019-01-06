"""
    Stack CatBoost (ensemble x8) predictions and LightGBM (single model) predictions
    Simply take a linear combination of the two, and tune the weight based on public leaderboard score
    We could get better performance if we train a meta-model on top of the predictions on a hold-out set
"""

import pandas as pd

lgb_single = pd.read_csv('submission/final_lgb_single.csv')
catboost_x8 = pd.read_csv('submission/final_catboost_ensemble_x8.csv')
print("Finished Loading the prediction results.")

weight = 0.7
stack = pd.DataFrame()
stack['ParcelId'] = lgb_single['ParcelId']
for col in ['201610', '201611', '201612', '201710', '201711', '201712']:
    stack[col] = weight * catboost_x8[col] + (1 - weight) * lgb_single[col]

print(stack.head())
stack.to_csv('submission/final_stack.csv', index=False)
