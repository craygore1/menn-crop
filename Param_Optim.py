import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from MENN import MENN
from sklearn.model_selection import KFold

StdTbl = pd.read_csv('CROPstd.csv')
StdTbl = StdTbl.dropna()
StdTbl['SubjDay'] = StdTbl['Subject'] + (StdTbl['Day']).astype(str)

subject_id_mapping = {id_: idx for idx, id_ in enumerate(sorted(StdTbl['Subject'].unique()))}
StdTbl['Subject_Id'] = StdTbl['Subject'].map(subject_id_mapping)

# Map day IDs to unique integers (nested within subjects)
day_id_mapping = {id_: idx for idx, id_ in enumerate(sorted(StdTbl['SubjDay'].unique()))}
StdTbl['Day_Id'] = StdTbl['SubjDay'].map(day_id_mapping)

StdTbl['Day_Id'] = StdTbl['SubjDay'].map(day_id_mapping)

X = StdTbl[['Time 0', 'Time 1', 'Time 2', 'Time 3', 'Time 4', 'Time 5'
    , 'Chrono', 'Odor_Int', 'Odor_PL', 'GAffect',
                      'GVigor', 'CompApp']].values
y = StdTbl[['Unhealthy', 'Fruit', 'Veggie', 'Percent_Cal', 'Protein', 'TFat',
                    'Percent_Carb', 'TSugar', 'Fiber', 'TFV', 'WGrain', 'RFDGrain',
                    'AddSugar']].values

resp_names = ['Unhealthy', 'Fruit', 'Veggie', 'Percent_Cal',
              'Protein', 'TFat', 'Percent_Carb', 'TSugar',
              'Fiber', 'TFV', 'WGrain', 'RFDGrain', 'AddSugar']

subject_ids = StdTbl['Subject_Id'].values
day_ids = StdTbl['Day_Id'].values

def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    wd = trial.suggest_loguniform('wd', 1e-5, 1e-2)
    dropout = trial.suggest_loguniform('dropout', 1e-2, 0.9)
    num_layers = trial.suggest_int('num_layers', 1, 5)  # Number of layers in the network
    hidden_dim = trial.suggest_int('hidden_dim', 8, 128)  # Neurons per hidden layer
    
    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)  # 5-fold CV
    fold_losses = []
    
    for train_idx, test_idx in kfolds.split(X):
        
        X_train, X_test = X[train_idx], X[test_idx] 
        y_train, y_test = y[train_idx], y[test_idx]
        train_subj_ids, test_subj_ids, train_day_ids, test_day_ids = subject_ids[train_idx], subject_ids[test_idx], day_ids[train_idx], day_ids[test_idx]
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        train_subj_ids = torch.tensor(train_subj_ids, dtype=torch.long)
        test_subj_ids = torch.tensor(test_subj_ids, dtype=torch.long)
        train_day_ids = torch.tensor(train_day_ids, dtype=torch.long)
        test_day_ids = torch.tensor(test_day_ids, dtype=torch.long)
    
        model = MENN(input_dim=X_train.shape[1], 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim, 
                             num_subjects=len(subject_id_mapping), 
                             num_days=len(day_id_mapping),
                             dropout=dropout)
        
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()
        
        
        epochs = 200
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train, train_subj_ids, train_day_ids)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
        
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test, test_subj_ids, test_day_ids)
            val_loss = criterion(val_predictions, y_test).item()
            
        fold_losses.append(val_loss)
        
        return sum(fold_losses) / len(fold_losses)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=150)

print(study.best_params)