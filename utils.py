import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def sort_columns_descending(matrix):
    # Count the number of non-zero elements in each column
    counts = np.count_nonzero(matrix, axis=0)

    # Get the indices that would sort the counts in descending order
    sorted_indices = np.argsort(counts)[::-1]

    # Sort the matrix columns using the sorted indices
    sorted_matrix = matrix[:, sorted_indices]

    # Calculate the current position of each column
    current_positions = np.zeros_like(sorted_indices)
    current_positions[sorted_indices] = np.arange(sorted_indices.size)

    return sorted_matrix, current_positions


def compare_np_arr(state1, state2):
    compare = (state1 == state2)
    flag = True
    size = compare.size
    compare = np.reshape(compare, (size, 1))
    n_rows, n_cols = compare.shape
    for i in range(n_rows):
        if not compare[i]:
            flag = False
            break

    return flag


def compareDisposition(a, b):
    compare = (a == b)
    flag = True
    n_rows, n_cols = compare.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if not compare[i, j]:
                flag = False
                break
    return flag


def compareState(state1, state2):
    state1_disposition = state1.disposition
    state2_disposition = state2.disposition
    return compare_np_arr(state1_disposition, state2_disposition)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        y = y.float()
        # Compute prediction and loss
        pred = model(X).float()
        loss = loss_fn(pred.squeeze(), y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
