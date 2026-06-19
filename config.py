# Updated features including the new historical margin metrics
FEATURES = [
    "Price Realization Rate",
    "Seasonality Index",
    "Total List Value", 
    "Actual Invoice Value",  
    "High-Value Product Mix %",  
    # "Historical Gross Margin (6M)", 
    # "Historical Gross Margin (12M)" , 
    "Dealer Target GM PCT" 
]

# Updated to match the business-friendly name used in the renaming script
TARGET = "Actual Gross Margin %"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.005,
    "subsample": 0.80,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "random_state": 42
}

# Updated path to the new final dataset
TRAINING_DATA_PATH = "ml_training_dataset.xlsx"
OUTPUT_EXCEL_PATH = "ml_prediction.xlsx"
MODEL_PIPELINE_PATH = "carrier_pricing_pipeline.joblib"