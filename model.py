import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import config

df = pd.read_excel(config.TRAINING_DATA_PATH, sheet_name="Sheet1")

historical_mask = df["Year"] < 2026
test_mask = df["Year"] == 2026

df_historical = df[historical_mask].copy()
df_test = df[test_mask].copy()

train_idx, val_idx = train_test_split(
    df_historical.index,
    test_size=0.20,
    random_state=42
)

X_train = df_historical.loc[train_idx, config.FEATURES]
y_train = df_historical.loc[train_idx, config.TARGET]

X_val = df_historical.loc[val_idx, config.FEATURES]
y_val = df_historical.loc[val_idx, config.TARGET]

X_test = df_test[config.FEATURES]
y_test = df_test[config.TARGET]

print("Training baseline Random Forest model...")
rf_model = RandomForestRegressor(**config.RF_PARAMS)
rf_model.fit(X_train, y_train)

rf_train_preds = rf_model.predict(X_train)
rf_val_preds = rf_model.predict(X_val)
rf_test_preds = rf_model.predict(X_test)

rf_train_r2 = r2_score(y_train, rf_train_preds)
rf_val_r2 = r2_score(y_val, rf_val_preds)
rf_test_r2 = r2_score(y_test, rf_test_preds)
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_preds))
rf_val_mae = mean_absolute_error(y_val, rf_val_preds)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_preds))
rf_test_mae = mean_absolute_error(y_test, rf_test_preds)

print("Training production XGBoost model...")
xgb_model = XGBRegressor(**config.XGB_PARAMS)
xgb_model.fit(X_train, y_train)

xgb_train_preds = xgb_model.predict(X_train)
xgb_val_preds = xgb_model.predict(X_val)
xgb_test_preds = xgb_model.predict(X_test)

xgb_train_r2 = r2_score(y_train, xgb_train_preds)
xgb_val_r2 = r2_score(y_val, xgb_val_preds)
xgb_test_r2 = r2_score(y_test, xgb_test_preds)
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))
xgb_val_mae = mean_absolute_error(y_val, xgb_val_preds)
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_preds))
xgb_test_mae = mean_absolute_error(y_test, xgb_test_preds)

rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": config.FEATURES,
    "RandomForest_Importance": rf_importances,
    "XGBoost_Importance": xgb_importances
}).sort_values(by="XGBoost_Importance", ascending=False).reset_index(drop=True)

print("\n" + "="*50)
print("BASELINE RANDOM FOREST EVALUATION REPORT")
print("="*50)
print(f"Training R2 Score   : {rf_train_r2:.4f}")
print(f"Validation R2 Score : {rf_val_r2:.4f}")
print(f"Testing R2 Score    : {rf_test_r2:.4f}")
print("-"*50)
print(f"Validation RMSE     : {rf_val_rmse:.4f}")
print(f"Validation MAE      : {rf_val_mae:.4f}")

print("\n" + "="*50)
print("PRODUCTION XGBOOST MODEL EVALUATION REPORT")
print("="*50)
print(f"Training R2 Score   : {xgb_train_r2:.4f}")
print(f"Validation R2 Score : {xgb_val_r2:.4f}")
print(f"Testing R2 Score    : {xgb_test_r2:.4f}")
print("-"*50)
print(f"Validation RMSE     : {xgb_val_rmse:.4f}")
print(f"Validation MAE      : {xgb_val_mae:.4f}")
print(f"Testing RMSE        : {xgb_test_rmse:.4f}")
print(f"Testing MAE         : {xgb_test_mae:.4f}")
print("-"*50)
print("FEATURE IMPORTANCE DISTRIBUTION (XGBOOST):")
for idx, row in importance_df.iterrows():
    print(f" - {row['Feature']:<40}: {row['XGBoost_Importance']:.4f}")
print("="*50 + "\n")

joblib.dump(xgb_model, config.MODEL_PIPELINE_PATH)

df["Predicted Gross Margin PCT"] = xgb_model.predict(df[config.FEATURES])
df["Predicted Gross Margin PCT"] = df["Predicted Gross Margin PCT"].round(4)

df["Predicted Invoice Price"] = df["Actual Invoice Value"].round(2)
df["Predicted COGS"] = (df["Predicted Invoice Price"] * (1.0 - df["Predicted Gross Margin PCT"])).round(2)

rf_metrics = pd.DataFrame({
    "Parameter": ["Last Trained Date", "Training R2", "Validation R2", "Validation RMSE", "Validation MAE", "Testing R2", "Testing RMSE", "Testing MAE"],
    "Value": [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(rf_train_r2, 4), round(rf_val_r2, 4), round(rf_val_rmse, 4), round(rf_val_mae, 4), round(rf_test_r2, 4), round(rf_test_rmse, 4), round(rf_test_mae, 4)]
})

xgb_metrics = pd.DataFrame({
    "Parameter": ["Last Trained Date", "Training R2", "Validation R2", "Validation RMSE", "Validation MAE", "Testing R2", "Testing RMSE", "Testing MAE"],
    "Value": [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(xgb_train_r2, 4), round(xgb_val_r2, 4), round(xgb_val_rmse, 4), round(xgb_val_mae, 4), round(xgb_test_r2, 4), round(xgb_test_rmse, 4), round(xgb_test_mae, 4)]
})

with pd.ExcelWriter(config.OUTPUT_EXCEL_PATH, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Predictions", index=False)
    xgb_metrics.to_excel(writer, sheet_name="XGBoost_Metrics", index=False)
    # rf_metrics.to_excel(writer, sheet_name="RandomForest_Metrics", index=False)
    importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

print(f"Production artifact successfully saved: {config.MODEL_PIPELINE_PATH}")
print(f"All updates saved cleanly inside sheets of: {config.OUTPUT_EXCEL_PATH}")