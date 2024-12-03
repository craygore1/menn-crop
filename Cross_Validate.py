import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from MENN import MENN
    
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NN, self).__init__()
        # Standard Feed-Forward Neural Net
        self.nn_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 13) 
        )
    
    def forward(self, X):
     
        x = self.nn_model(X)
        
        return x
    

StdTbl = pd.read_csv('CROPstd.csv')
StdTbl = StdTbl.dropna()
StdTbl['SubjDay'] = StdTbl['Subject'] + (StdTbl['Day']).astype(str)

subject_id_mapping = {id_: idx for idx, id_ in enumerate(sorted(StdTbl['Subject'].unique()))}
StdTbl['Subject_Id'] = StdTbl['Subject'].map(subject_id_mapping)

# Map day IDs to unique integers (nested within subjects)
day_id_mapping = {id_: idx for idx, id_ in enumerate(sorted(StdTbl['SubjDay'].unique()))}
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



def cross_validate(X, y, subject_ids, day_ids, num_layers = 1, hidden_dim=10, k=5, epochs=100, lr=0.01, weight_decay = 1e-2):
    kfolds = KFold(n_splits=k, shuffle=True, random_state=3)
    menn_mse, nn_mse = [],[]
    
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
        
        
        num_subjects = len(np.unique(subject_ids))
        num_days = len(np.unique(day_ids))
        
        menn = MENN(input_dim=X_train.shape[1], 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim, 
                             num_subjects=num_subjects, 
                             num_days=num_days)
        
        nn_model = NN(input_dim=X_train.shape[1], hidden_dim = hidden_dim)
        
        menn_criterion = nn.MSELoss()
        nn_criterion = nn.MSELoss()
        
        menn_optimizer = optim.Adam(menn.parameters(), lr=lr, weight_decay=weight_decay)
        nn_optimizer = optim.Adam(nn_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # MENN Training
        for epoch in range(epochs):
            menn.train()
            menn_optimizer.zero_grad()
            menn_predictions = menn(X_train, train_subj_ids, train_day_ids)
            menn_loss = menn_criterion(menn_predictions, y_train)
            menn_loss.backward()
            menn_optimizer.step()
            
        
        # MENN Evaluation
        menn.eval()
        with torch.no_grad():
            menn_predictions = menn(X_test, test_subj_ids, test_day_ids)
            menn_mse.append(menn_criterion(menn_predictions, y_test).numpy())
        
        # NN Training
        for epoch in range(epochs):
            nn_model.train()
            nn_optimizer.zero_grad()
            nn_predictions = nn_model(X_train)
            nn_loss = nn_criterion(nn_predictions, y_train)
            nn_loss.backward()
            nn_optimizer.step()
            
        
        # NN Evaluation
        nn_model.eval()
        with torch.no_grad():
            nn_predictions = nn_model(X_test)
            nn_mse.append(nn_criterion(nn_predictions, y_test).numpy())        
    
    results = {
        "MENN": {
            "MSE": np.mean(menn_mse),
        },
        "Standard NN": {
            "MSE": np.mean(nn_mse),
        }
    }
    return menn_mse, nn_mse, results

menn_mse, nn_mse, results = cross_validate(X, y, subject_ids, day_ids, num_layers=1, hidden_dim=10,
                                               k=10, epochs=200, lr=0.0601, weight_decay=0.0057)

print(results)
