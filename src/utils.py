# Colours for plots
COLOURS = ['g', 'blue', 'r', 'c', 'm', 'k']

# Hyperparameters for eval1 pipeline 
EVAL1_Cora = {'args_p': 0.013,
              'args_c': 0.003,
              'args_binary': True}

EVAL1_PubMed = {'args_p': 0.1,
                'args_c': 0.0005,
                'args_binary': False}

Cora_class_distrib = [0.135, 0.075, 0.158, 0.300, 0.158,
                       0.109, 0.063]

# Model structure hyperparameters for Cora dataset, GCN model
hparams_Cora_GCN = {
    'hidden_dim': [16],
    'dropout': 0.5
}

# Training hyperparameters for Cora dataset, GCN model
params_Cora_GCN = {
    'num_epochs': 50,
    'lr': 0.01,
    'wd': 5e-4
}

# Cora - GAT
hparams_Cora_GAT = {
    'hidden_dim': [8],
    'dropout': 0.6,
    'n_heads': [8, 1]
}

params_Cora_GAT = {
    'num_epochs': 100,
    'lr': 0.005,
    'wd': 5e-4
}

# PubMed - GCN
hparams_PubMed_GCN = hparams_Cora_GCN
params_PubMed_GCN = {
    'num_epochs': 150,
    'lr': 0.01,
    'wd': 5e-4
}

# PubMed - GAT
hparams_PubMed_GAT = {
    'hidden_dim': [8],
    'dropout': 0.6,
    'n_heads': [8, 8]
}

params_PubMed_GAT = {
    'num_epochs': 250,
    'lr': 0.005,
    'wd': 5e-4
}


### Synthethic datasets

# Model structure hyperparameters for syn dataset, GCN model
hparams_syn1_GCN = {
    'hidden_dim': [20, 20, 20],
  		'dropout': 0
}

# Training hyperparameters for syn dataset, GCN model
params_syn1_GCN = {
	'num_epochs': 1000,
	'lr': 0.001,
	'wd': 5e-3
}
