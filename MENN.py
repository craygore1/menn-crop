import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class MENN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, num_subjects, num_days, dropout=0.1):
        super(MENN, self).__init__()
        # Neural network for fixed effects
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 13)) 
        self.fixed_nn = nn.Sequential(*layers)
        
        # Subject-level random effects
        self.subject_effects = nn.Embedding(num_subjects, 13)
        
        # Day-level random effects
        self.day_effects = nn.Embedding(num_days, 13)

    def forward(self, X, subject_ids, day_ids):
        # Fixed effects prediction
        fixed_effects = self.fixed_nn(X)
        
        # Random effects contributions
        subject_random_effects = self.subject_effects(subject_ids).squeeze()
        day_random_effects = self.day_effects(day_ids).squeeze()
        
        return fixed_effects.squeeze() + subject_random_effects + day_random_effects
