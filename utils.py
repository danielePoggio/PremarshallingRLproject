import copy
import time
import numpy as np
import matplotlib.pyplot as plt


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
    state1 = state1.disposition
    state2 = state2.disposition
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
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])  # azione necessaria perchè l'agentNN2 veda
    # l'ambiente
    tot_cost = 0
    for t in range(time_limit):
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        decision = []  # siccome l'agentNN2 non prende decisioni è come se la lista delle decisioni fosse vuota
        action = agente.get_action(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = enviroment.step(decision + action)
        agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
        # print(enviroment.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    return tot_cost


def marshallingWithAgentNN(enviroment, agente, time_limit, iterations):
    start_time = time.time()
    agente.learnFrequency(num_episode=1)
    # Eseguiamo apprendimento dell'agentNN2
    agente.learn(iterations=iterations)
    # Resettiamo ambiente
    obs = enviroment.reset()
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
    tot_cost = 0
    no_empty_decisions = 0
    # Partiamo con le iterazioni
    for t in range(time_limit):
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        decision = agente.agentDecisionRandom(grid=agente.actualDisposition)  # prende decisione
        if decision != []:
            no_empty_decisions += 1
        obs['actual_warehouse'].disposition = copy.deepcopy(agente.actualDisposition.disposition)
        action = agente.get_action(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = enviroment.step(decision + action)
        agente.actualDisposition.disposition = copy.deepcopy(obs['actual_warehouse'].disposition)
        tot_cost += cost
        # print(enviroment.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return tot_cost, elapsed_time, no_empty_decisions


def marshallingWithAgentNN2(enviroment, agente, time_limit, iterations):
    start_time = time.time()
    agente.learnFrequency(num_episode=1)
    # Eseguiamo apprendimento dell'agentNN2
    agente.learn(iterations=iterations)
    # Resettiamo ambiente
    obs = enviroment.reset()
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
    tot_cost = 0
    no_empty_decisions = 0
    # Partiamo con le iterazioni
    for t in range(time_limit):
        # Stampiamo ordini e nuovi arrivi
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        # Agente prende decisione in base allo stato che osserva
        decision = agente.agentDecision(grid=copy.deepcopy(agente.actualDisposition))  # prende decisione
        if decision != []:
            no_empty_decisions += 1
        # Aggiorno obs per farlo funzionare in get_action
        obs['actual_warehouse'] = copy.deepcopy(agente.actualDisposition)
        action = agente.get_action(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = enviroment.step(decision + action)
        # Agente osserva nuova disposizione per prendere decisione futura
        agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
        # print(enviroment.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return tot_cost, elapsed_time, no_empty_decisions


def marshallingWithAgentPD(enviroment, agente, time_limit, iterations):
    start_time = time.time()
    agente.learnFrequency(num_episode=1)
    # Eseguiamo apprendimento dell'agentNN2
    agente.learn(iterations=iterations)
    # Resettiamo ambiente
    obs = enviroment.reset()
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
    tot_cost = 0
    no_empty_decisions = 0
    # Partiamo con le iterazioni
    for t in range(time_limit):
        # Agente prende decisione in base allo stato che osserva
        decision = agente.agentDecision(grid=copy.deepcopy(agente.actualDisposition))  # prende decisione
        if decision != []:
            no_empty_decisions += 1
        # Aggiorno obs per farlo funzionare in get_action
        obs['actual_warehouse'] = copy.deepcopy(agente.actualDisposition)
        action = agente.get_action(obs=obs)  # risolve ordini
        obs, cost, info = enviroment.step(decision + action)
        # Agente osserva nuova disposizione per prendere decisione futura
        agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
    print(tot_cost)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return tot_cost, elapsed_time, no_empty_decisions


def plot_2d_graph(x_data, y_data, x_label, y_label, title):
    # Creazione del grafico
    plt.plot(x_data, y_data)

    # Titolo del grafico
    plt.title(title)

    # Etichette degli assi
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Visualizzazione del grafico
    plt.show()


def plot_2(x, y1, y2, x_label, y_label, title, path):
    plt.figure()
    plt.plot(x, y1, label="AgentNN1")
    plt.plot(x, y2, label="AgentNN2")
    # Titolo del grafico
    plt.title('')

    # Etichette degli assi
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Mostra la legenda
    plt.legend()

    # Visualizzazione del grafico
    save_file_path = path + '/' + title + '.png'
    plt.savefig(save_file_path)
    plt.close()


def plot_3(x_data, yNoLearn, yNN, yNN2, x_label, y_label, title, path):
    # Creazione del grafico
    plt.figure()
    plt.plot(x_data, yNoLearn, label="Agent No learning")
    plt.plot(x_data, yNN, label="AgentNN1")
    plt.plot(x_data, yNN2, label="AgentNN2")

    # Titolo del grafico
    plt.title('')

    # Etichette degli assi
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Mostra la legenda
    plt.legend()

    # Visualizzazione del grafico
    save_file_path = path + '/' + title + '.png'

    plt.savefig(save_file_path)
    plt.close()

