import copy

import numpy as np


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


def restore_original_order(sorted_matrix, current_positions):
    # Get the indices that would restore the original order
    original_indices = np.argsort(current_positions)

    # Restore the original order of the columns
    original_matrix = sorted_matrix[:, original_indices]

    # Update the current positions of the columns
    current_positions[original_indices] = np.arange(original_indices.size)

    return original_matrix, current_positions


def transformedAction(action, current_positions):
    for move in action:
        move['col1'] = current_positions[move['col1']]
        move['col2'] = current_positions[move['col2']]
    return action


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
    # state1 = state1.disposition
    # state2 = state2.disposition
    return compare_np_arr(state1, state2)


def findNonZeroElem(matrice, colonna):
    for riga, elemento in enumerate(matrice):
        if elemento[colonna] != 0:
            return riga
    return -1


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


def marshallingWithoutAgent(enviroment, agente, time_limit):
    obs = enviroment.reset()
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])  # azione necessaria perchè l'agente veda
    # l'ambiente
    tot_cost = 0
    for t in range(time_limit):
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        decision = []  # siccome l'agente non prende decisioni è come se la lista delle decisioni fosse vuota
        action = agente.get_action(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = enviroment.step(decision + action)
        agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
        print(enviroment.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    return tot_cost


def marshallingWithAgent(enviroment, agente, time_limit):
    agente.learnFrequency(num_episode=1)
    # Eseguiamo apprendimento dell'agente
    agente.learn(iterations=10)
    # resettiamo ambiente
    obs = enviroment.reset()
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
    tot_cost = 0
    # Partiamo con le iterazioni
    for t in range(time_limit):
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        decision = agente.agentDecision(grid=agente.actualDisposition, probStop=0.1)  # prende decisione
        action = agente.get_action(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = enviroment.step(decision + action)
        agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
        print(enviroment.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    return tot_cost


def testingParameter(parameter, test_value):
