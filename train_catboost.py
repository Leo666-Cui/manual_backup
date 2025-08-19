import json
import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def improved_train_and_evaluate():
    """
    This function implements an improved ML workflow:
    1. Loads data.
    2. Uses Stratified K-Fold Cross-Validation for robust evaluation.
    3. Handles class imbalance using 'scale_pos_weight'.
    4. Performs Hyperparameter Tuning with GridSearchCV to find the best model.
    """
    # --- Step 1: Load and preprocess the data ---
    try:
        with open('multi_agent_results.json', 'r') as f:
            features_data = json.load(f)
        with open('HCC_label.json', 'r') as f:
            labels_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the JSON files are in the same directory.")
        return

    if features_data:  # 确保 features_data 不为空
        # 获取第一个病人的ID
        first_patient_id = list(features_data.keys())[0]
        # 获取该病人拥有的 pattern 数量
        num_patterns = len(features_data[first_patient_id])
    else:
        num_patterns = 0
    print(f"一共有{num_patterns}个问题")

    records = []
    for patient_id, patterns in features_data.items():
        if patient_id in labels_data:
            sorted_keys = sorted(patterns.keys())
            answers = [patterns[key]['answer'] for key in sorted_keys]
            record = {'patient_id': patient_id, 'label': labels_data[patient_id]}
            for i, answer in enumerate(answers):
                record[f'feature_{i+1}'] = answer
            records.append(record)
    print(record)

    df = pd.DataFrame(records)
    if df.empty:
        print("Error: No matching patient data found.")
        return

    X = df.drop(['patient_id', 'label'], axis=1)
    y = df['label']

    print("--- Data Loading ---")
    print(f"Successfully loaded data for {len(df)} patients.")
    print(f"Class distribution in the dataset:\n{y.value_counts()}")
    
    # --- Step 2: Set up the Improved Workflow ---
    
    # a) Calculate scale_pos_weight for handling imbalance
    # This is calculated once on the entire dataset and passed to the model.
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
    print(f"\nCalculated 'scale_pos_weight' to handle imbalance: {scale_pos_weight:.2f}\n")

    # b) Define the model and the parameter grid for GridSearchCV
    # These are the parameters we want to tune.
    # XGBoost
    """param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    # c) Initialize the XGBoost classifier with the imbalance handler
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight, # Crucial for imbalance
        random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # e) Set up GridSearchCV
    # This will automatically search the best parameters using 5-fold CV.
    # We use 'f1' as the scoring metric because it's better for imbalanced data.
    grid_search = GridSearchCV(
        estimator=xgb_classifier, 
        param_grid=param_grid, 
        scoring='f1', 
        cv=cv, 
        n_jobs=-1, # Use all available CPU cores
        verbose=2
    )
    """

    # catboost
    # param_grid = {
    # 'depth': [4, 6, 8],          # 对应 'max_depth'
    # 'learning_rate': [0.01, 0.05, 0.1],
    # 'iterations': [100, 200, 300], # 对应 'n_estimators'
    # 'rsm': [0.8, 1.0]                # 对应 'colsample_bytree'
    # }

    param_grid = {
    'depth': [4], 
    'learning_rate': [0.01], 
    'iterations': [100], 
    'rsm': [0.8]
    }

    cat_classifier = CatBoostClassifier(
    #task_type='GPU',
    loss_function='Logloss',      # 对应 objective
    scale_pos_weight=scale_pos_weight,
    random_seed=42,               # 对应 random_state
    verbose=0                     # 新增：让CatBoost在GridSearch时保持“沉默”
    )

    # d) Set up Stratified K-Fold Cross-Validation
    # n_splits=5 means 5-fold cross-validation. shuffle=True is important.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=cat_classifier, 
        param_grid=param_grid, 
        scoring='f1', 
        cv=cv, 
        n_jobs=1,
        verbose=2
    )

    print("--- Starting GridSearchCV for Hyperparameter Tuning ---")
    print("This may take a few moments...")
    
    # --- Step 3: Run the Training and Tuning ---
    grid_search.fit(X, y)

    print("\n--- Tuning Complete ---")
    print(f"Best F1-score found during search: {grid_search.best_score_:.4f}")
    print("Best parameters found:")
    print(grid_search.best_params_)

    # --- Step 4: Evaluate the Best Model Found ---
    # We use the best estimator found by GridSearchCV for final evaluation.
    best_model = grid_search.best_estimator_
    
    # To get a final report, we can make predictions on the whole dataset
    # (This shows how well the final model fits the data it was trained on)
    y_pred_full = best_model.predict(X)
    
    print("\n--- Final Evaluation Report (on full dataset) ---")
    print(f"Accuracy: {accuracy_score(y, y_pred_full):.4f}")
    overall_auc = roc_auc_score(y, y_pred_full)
    print(f"整体 AUC (Overall AUC): {overall_auc:.4f}")


    print("\nClassification Report:")
    print(classification_report(y, y_pred_full))

    print("\nConfusion Matrix (sum of predictions on full dataset):")
    cm = confusion_matrix(y, y_pred_full)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Final Confusion Matrix on Full Data')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    # plt.savefig('improved_confusion_matrix.png')
    # print("Improved confusion matrix plot has been saved as 'improved_confusion_matrix.png'")


    """# 定义要保存的文件名
    model_filename = 'catboost_best_model.joblib'

    # 使用 joblib.dump 保存 best_model 对象
    print(f"\n--- Saving the best model to {model_filename} ---")
    joblib.dump(best_model, model_filename)
    print("Model saved successfully.")"""

if __name__ == '__main__':
    improved_train_and_evaluate()
