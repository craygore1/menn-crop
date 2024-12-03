import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from MENN import MENN
from sklearn.model_selection import train_test_split

StdTbl = pd.read_csv('CROPstd.csv')
StdTbl = StdTbl.dropna()
StdTbl['SubjDay'] = StdTbl['Subject'] + (StdTbl['Day']).astype(str)

subject_id_mapping = {id_: idx for idx, id_ in enumerate(sorted(StdTbl['Subject'].unique()))}
StdTbl['Subject_Id'] = StdTbl['Subject'].map(subject_id_mapping)

# Map day IDs to unique integers (nested within subjects)
day_id_mapping = {id_: idx for idx, id_ in enumerate(sorted(StdTbl['SubjDay'].unique()))}
StdTbl['Day_Id'] = StdTbl['SubjDay'].map(day_id_mapping)

train_data, test_data = train_test_split(StdTbl, test_size=0.2, stratify=StdTbl['Subject_Id'], random_state=3)

X_train = train_data[['Time 0', 'Time 1', 'Time 2', 'Time 3', 'Time 4', 'Time 5'
    , 'Chrono', 'Odor_Int', 'Odor_PL', 'GAffect',
                      'GVigor', 'CompApp']].values

y_train = train_data[['Unhealthy', 'Fruit', 'Veggie', 'Percent_Cal', 'Protein', 'TFat',
                    'Percent_Carb', 'TSugar', 'Fiber', 'TFV', 'WGrain', 'RFDGrain',
                    'AddSugar']].values

X_test = test_data[['Time 0', 'Time 1', 'Time 2', 'Time 3', 'Time 4', 'Time 5'
    , 'Chrono', 'Odor_Int', 'Odor_PL', 'GAffect',
                      'GVigor', 'CompApp']].values

y_test = test_data[['Unhealthy', 'Fruit', 'Veggie', 'Percent_Cal', 'Protein', 'TFat',
                    'Percent_Carb', 'TSugar', 'Fiber', 'TFV', 'WGrain', 'RFDGrain',
                    'AddSugar']].values

train_subject_ids = train_data['Subject_Id'].values
train_day_ids = train_data['Day_Id'].values
test_subject_ids = test_data['Subject_Id'].values
test_day_ids = test_data['Day_Id'].values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_subject_ids_tensor = torch.tensor(train_subject_ids, dtype=torch.long)
train_day_ids_tensor = torch.tensor(train_day_ids, dtype=torch.long)
test_subject_ids_tensor = torch.tensor(test_subject_ids, dtype=torch.long)
test_day_ids_tensor = torch.tensor(test_day_ids, dtype=torch.long)

def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    wd = trial.suggest_loguniform('wd', 1e-5, 1e-2)
    num_layers = trial.suggest_int('num_layers', 1, 5)  # Number of layers in the network
    hidden_dim = trial.suggest_int('hidden_dim', 8, 128)  # Neurons per hidden layer
    
    
    model = MENN(input_dim=X_train_tensor.shape[1], 
                         num_layers=num_layers, 
                         hidden_dim=hidden_dim, 
                         num_subjects=len(subject_id_mapping), 
                         num_days=len(day_id_mapping))
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    
    
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor, train_subject_ids_tensor, train_day_ids_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_test_tensor, test_subject_ids_tensor, test_day_ids_tensor)
        val_loss = criterion(val_predictions, y_test_tensor).item()
    
    return val_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(study.best_params)