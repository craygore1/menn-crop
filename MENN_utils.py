# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def model_train(model, X_train, y_train, subj_ids, day_ids, epochs, lr, weight_decay):
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    #Training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train, subj_ids, day_ids)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
def model_eval(model, X_test, y_test, test_subj_ids, test_day_ids):
    # MENN Evaluation
    menn.eval()
    with torch.no_grad():
        menn_predictions = menn(X_test, test_subj_ids, test_day_ids)
        menn_mse.append(menn_criterion(menn_predictions, y_test).numpy())
