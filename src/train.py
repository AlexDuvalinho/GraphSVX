""" train.py

	Trains our model 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from utils.graph_utils import GraphSampler
from torch.autograd import Variable


##################################################################
# Node Classification on real world datasets
##################################################################

def train_and_val(model, data, num_epochs, lr, wd, verbose=True):
    """ Model training

    Args:
        model (pyg): model trained, previously defined
        data (torch_geometric.Data): dataset the model is trained on
        num_epochs (int): number of epochs 
        lr (float): learning rate
        wd (float): weight decay
        verbose (bool, optional): print information. Defaults to True.

    """

    # Define the optimizer for the learning process
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Training and eval modes
    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    best = np.inf
    bad_counter = 0

    for epoch in tqdm(range(num_epochs), desc='Training', leave=False):
        if epoch == 0:
            print('       |     Trainging     |     Validation     |')
            print('       |-------------------|--------------------|')
            print(' Epoch |  loss    accuracy |  loss    accuracy  |')
            print('-------|-------------------|--------------------|')

        # Training
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        train_loss = F.nll_loss(
            output[data.train_mask], data.y[data.train_mask])
            
        #train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        
        # Store results
        train_loss_values.append(train_loss.item())
        train_acc_values.append(train_acc.item())

        # Validation
        val_loss, val_acc = evaluate(data, model, data.val_mask)
        val_loss_values.append(val_loss.item())
        val_acc_values.append(val_acc.item())

        if val_loss_values[-1] < best:
            bad_counter = 0
            log = '  {:3d}  | {:.4f}    {:.4f}  | {:.4f}    {:.4f}   |'.format(epoch + 1,
                                                                               train_loss.item(),
                                                                               train_acc.item(),
                                                                               val_loss.item(),
                                                                               val_acc.item())
        
            if verbose:
                tqdm.write(log)

            best = val_loss_values[-1]
        else:
            bad_counter += 1

    print('-------------------------------------------------')


def evaluate(data, model, mask):
    """ Model evaluation on validation data

    Args: 
            mask (torch.tensor): validation mask

    Returns:
            [torch.Tensor]: loss function's value on validation data
            [torch.Tensor]: accuracy of model on validation data
    """
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[mask], data.y[mask])
        acc = accuracy(output[mask], data.y[mask])

    return loss, acc


def accuracy(output, labels):
    """ Computes accuracy metric for the model on test set

    Args:
            output (tensor): class predictions for each node, computed with our model
            labels (tensor): true label of each node

    Returns:
            [tensor]: accuracy metric

    """
    # Find predicted label from predicted probabilities
    _, pred = output.max(dim=1)
    # Derive number of correct predicted labels
    correct = pred.eq(labels).double()
    # Sum over all nodes
    correct = correct.sum()

    # Return accuracy metric
    return correct / len(labels)


##################################################################
# Node Classification on synthetic datasets
##################################################################

def train_syn(data, model, args):
    """ Train model on synthetic datasets, for node classification
    """
    # Optimizer 
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training 
    for epoch in range(args.num_epochs):
        total_loss = 0
        model.train()
        opt.zero_grad()

        pred = model(data.x, data.edge_index)
        pred = pred[data.train_mask]
        label = data.y[data.train_mask]

        loss = F.nll_loss(pred, label)
        #loss = model.loss(pred, label)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        opt.step()
        total_loss += loss.item() * 1
        
        # Evaluate on validation set
        if epoch % 10 == 0:
            train_acc = test(data, model, data.train_mask)
            val_acc = test(data, model, data.val_mask)
            print("Epoch {}. Loss: {:.4f}. Train accuracy: {:.4f}. Val accuracy: {:.4f}".format(
                epoch, total_loss, train_acc, val_acc))
    total_loss = total_loss / data.x.shape[0]


def test(data, model, mask):
    """ Evaluate model performance 
    For node and graph classification tasks

    Args: 
        mask (tensor): indicate train, validation or test set

    Returns:
        int: accuracy score 
    """
    # Eval mode
    model.eval()
    correct = 0

    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        pred = pred.argmax(dim=1)
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]
        correct += pred.eq(label).sum().item()

    total = (mask == True).nonzero().shape[0]
    return correct / total


##################################################################
# Graph Classification 
##################################################################


def train_gc(data, model, args):
    """ Train model for node classification tasks
    """
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Process input and pass to DataLoader to batch it 
    dataset_sampler = GraphSampler(data)
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True
        )
    
    # Training
    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        model.train()
        for batch_idx, df in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            y_pred = model(df["feats"], df["adj"])
            loss = F.nll_loss(y_pred, df['label'])
            #loss = model.loss(y_pred, df['label'])
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            avg_loss += loss

        # Evaluation
        avg_loss /= batch_idx + 1
        if epoch % 10 == 0:
            train_acc = test(data, model, data.train_mask)
            val_acc = test(data, model, data.val_mask)
            print("Epoch {}. Loss: {:.4f}. Train accuracy: {:.4f}. Val accuracy: {:.4f}".format(
                epoch, avg_loss, train_acc, val_acc))



