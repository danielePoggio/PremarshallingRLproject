import numpy as np
from tqdm import tqdm
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from itertools import permutations
from utils import compareState, train_loop


class Net(nn.Module):

    def __init__(self, n_rows, n_cols, n_parcel_items):
        super(Net, self).__init__()
        # n_parcel_items input channels, n_parcel_items output channels, 3x3 square convolution kernel
        self.conv = nn.Conv2d(n_parcel_items, n_parcel_items, kernel_size=3, padding=1)
        # n_parcel_items*n_rows*n_cols input channels, n_rows*n_cols output channels
        self.fc1 = nn.Linear(n_parcel_items * n_rows * n_cols, n_rows * n_cols)
        # n_rows*n_cols input channels, 1 output channel
        self.fc3 = nn.Linear(n_rows * n_cols, 1)

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        x = torch.flatten(x, start_dim=1)
        x = F.sigmoid(self.fc1(x))
        x = self.fc3(x)
        return x


class StatePDDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.data = X
        self.target = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         y = y.float()
#         # Compute prediction and loss
#         pred = model(X).float()
#         loss = loss_fn(pred.squeeze(), y)
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


class AgentNN:
    def __init__(self, warehouse, alpha, gamma, n_item, time_limit, eps=0.1):
        self.stateList = []
        self.valueFunction = []
        self.actualDisposition = None
        self.eps = eps
        self.time_limit = time_limit
        self.n_explored_state = 0
        self.n_cols = warehouse.n_cols
        self.n_rows = warehouse.n_rows
        self.n_item = n_item
        self.n_moves = 1
        self.n_time = 0
        self.env = copy.deepcopy(warehouse)
        self.learningCoeff = alpha
        self.discountCoeff = gamma
        self.modelNN = Net(n_rows=self.n_rows, n_cols=self.n_cols, n_parcel_items=self.n_item)
        self.parcelObs = np.zeros(n_item)
        self.parcelFreq = np.zeros(n_item)
        self.tot_parcel = 0
        self.Q_Factor = []  # lista di dizionari tipo
        # {'state' : disposition, 'decisionsKnown' : [], 'bestDecision' : decision,
        # 'bestQ': q}
        # dove la lista in decisionsKnown è composta da dizionari del tipo {'decision': decision, 'Q_factor': Q}
        # SERVE PER RIDURRE TEMPO DI RICERCA COPPIA STATI-AZIONE

    def get_action(self, obs):  # aggiornare sempre cosa vede l'agentNN2!
        self.n_time += 1
        act = []
        #
        # SATISFY ORDERS:
        for i, order in enumerate(obs['order']):
            # get first position of in which there is an ordered block
            pos = np.transpose(
                np.where(self.actualDisposition.disposition == order)
            )[0]
            row, col = pos[0], pos[1]
            for ii in range(0, row):
                if self.actualDisposition.disposition[ii, col] != 0:
                    # sposto gli oggetti nella prima colonna libera
                    for c in range(self.actualDisposition.n_cols - 1, -1, -1):  # esploro le varie colonne
                        if self.actualDisposition.disposition[0, c] == 0:  # trovato colonna libera
                            act.append(
                                {
                                    'type': 'P',
                                    'col1': col,
                                    'col2': c,
                                }
                            )
                            self.actualDisposition._move(col1=col, col2=c)
                            # r = 1
                            # while self.actualDisposition.disposition[r, c] == 0 and r < obs[
                            #     'actual_warehouse'].n_rows:  # mi salvo posizione riga
                            #     if r == self.actualDisposition.n_rows - 1:  # sono arrivato in fondo
                            #         break
                            #     r += 1
                            # self.actualDisposition.disposition[r, c] = self.actualDisposition.disposition[ii, col]
                            # self.actualDisposition.disposition[ii, col] = 0
            act.append(
                {'type': 'O', 'col': col, 'n_order': i}
            )
            self.actualDisposition._take(col=col)

        # LOCATE NEW PARCELS
        for i, parcel in enumerate(obs['new_parcel']):
            for col in range(self.actualDisposition.n_cols):
                if self.actualDisposition.disposition[0, col] == 0:
                    act.append(
                        {'type': 'N', 'n_parcel': i, 'col': col}
                    )
                    self.actualDisposition._locate(
                        parcel,
                        col
                    )
                    break
        # plt.imshow(obs['actual_warehouse'].disposition)
        return act

    def learnFrequency(self, num_episode=1):
        obs = self.env.reset()
        self.actualDisposition = obs['actual_warehouse']
        for _ in tqdm(range(num_episode * self.time_limit)):
            # print('Order:', obs['order'])
            # print('New Parcel:', obs['new_parcel'])
            # obs['actual_warehouse'] = agent.actualDisposition
            action = self.get_action(obs=obs)
            # print(action)
            obs, cost, info = self.env.step(action)
            for parcel in obs['order']:
                self.parcelObs[parcel - 1] += 1
                self.tot_parcel += 1

        self.parcelFreq = self.parcelObs / self.tot_parcel

    def defineState(self, state_disposition):
        return copy.deepcopy(state_disposition)

    def explore_decision(self):
        choice = np.random.binomial(1, p=self.eps)
        if choice == 1:
            decision = self.exploration()
        else:
            decision = self.exploitation()
        self.actualDecision = decision
        return decision

    def exploration(self):
        #  Supponiamo di fare un solo movimento
        decision = []
        number_mov = 1
        n_rep = 0
        n_action = 0
        while n_action < number_mov:
            n_rep += 1
            if n_rep > 1000:
                print("HELP")
            col1 = np.random.randint(0, self.n_cols)
            col2 = np.random.randint(0, self.n_cols)
            if self.actualDisposition.disposition[0, col2] == 0 and col1 != col2:
                # eseguo azione se le colonne non sono identiche e se la colonna 2 non è piena e se la colonna 1 non
                # è vuota
                decision.append(
                    {'type': 'M', 'col1': col1, 'col2': col2}
                )
                self.actualDisposition._move(
                    col1,
                    col2
                )
                n_action += 1
                if n_rep == 900:
                    decision == []
                    break
        return decision

    def exploitation(self):
        decision = []
        state = self.defineState(self.actualDisposition)
        # allora devo valutare il minimo tra i Q-Factor
        findState = 0
        q_factor = self.Q_Factor
        for dictQFactor in q_factor:
            # guardare bene l'if perchè tratta vettori di lunghezza diversa!!
            if compareState(dictQFactor['state'],
                            self.actualDisposition):  # ho trovato lo stato
                findState = 1
                changeDisposition = dictQFactor[
                    'bestDecision']  # selezione l'azione migliore che conosco in quello stato
                for action in changeDisposition:
                    if self.actualDisposition.disposition[0, action['col2']] == 0:
                        col1 = action['col1']
                        col2 = action['col2']

                        self.actualDisposition._move(
                            col1,
                            col2
                        )
                        decision.append(action)
                    else:
                        break

                break
        if findState == 0:  # non ho trovato lo stato
            decision = self.exploration()

        return decision

    def updateQValuePD(self, old, decision, reward, new):  # aggiorno Q_Factor
        q_factor = self.Q_Factor
        alpha = self.learningCoeff
        gamma = self.discountCoeff
        # setto gli indicatori degli stati a -1, così nel caso la coppia (stato, azione) non fosse già stata visitata,
        # la creo io -> in pratica fanno anche da flag
        indexOldState = -1
        indexNewState = -1
        indexAction = -1
        j = 0
        equalState = 0
        if compareState(old, new):
            equalState = 1
        for q in q_factor:
            if compareState(q['state'], old):  # lo stato era già stato visitato
                indexOldState = j
                i = 0
                for act in q['decisionsKnown']:
                    if act['decision'] == decision:  # la coppia stato-azione era già stata visitata-> allora esiste un
                        # Q-factor
                        indexAction = i
                    i = i + 1
                if equalState == 1:  # non ha senso andare avanti con il ciclo for
                    break
            if compareState(q['state'], new):  # stato già visitato
                indexNewState = j
            if (indexOldState != -1) and (indexNewState != -1):  # esco dal ciclo for perchè ho trovato
                # gli stati che mi servivano
                break
            j = j + 1

        # In base ai risultati della ricerca vado ad aggiornare self.Q_Factor
        if indexOldState != -1 and indexAction != -1 and indexNewState != -1:  # tutti gli stati sono stati visitati
            Q_new = self.Q_Factor[indexNewState]['bestQ']
            Q_old = self.Q_Factor[indexOldState]['decisionsKnown'][indexAction]['Q_factor']
            Q_old = (1 - alpha) * Q_old + alpha * (reward + gamma * Q_new)
            # ora aggiorno valore Q(state,action)
            self.Q_Factor[indexOldState]['decisionsKnown'][indexAction]['Q_factor'] = Q_old
            # update bestQ
            if self.Q_Factor[indexOldState]['bestQ'] > Q_old:
                self.Q_Factor[indexOldState]['bestDecision'] = decision
                self.Q_Factor[indexOldState]['bestQ'] = Q_old
        elif indexOldState == -1 and indexNewState == -1:
            self.n_explored_state += 1
            # creo lo stato
            Q_old = alpha * reward
            self.Q_Factor.append(
                {
                    'state': old,
                    'decisionsKnown': [{'decision': decision, 'Q_factor': Q_old}],
                    'bestDecision': decision,
                    'bestQ': Q_old
                }
            )
        elif indexOldState == -1 and indexNewState != -1:
            self.n_explored_state += 1
            Q_new = self.Q_Factor[indexNewState]['bestQ']
            Q_old = alpha * (reward + gamma * Q_new)
            # creo lo stato
            self.Q_Factor.append(
                {
                    'state': old,
                    'decisionsKnown': [{'decision': decision, 'Q_factor': Q_old}],
                    'bestDecision': decision,
                    'bestQ': Q_old
                }
            )
        elif indexOldState != -1 and indexAction != -1 and indexNewState == -1:
            Q_old = self.Q_Factor[indexOldState]['decisionsKnown'][indexAction]['Q_factor']
            Q_old = (1 - alpha) * Q_old + alpha * reward
            # ora aggiorno valore Q(state,action)
            self.Q_Factor[indexOldState]['decisionsKnown'][indexAction]['Q_factor'] = Q_old
            # update bestQ
            if self.Q_Factor[indexOldState]['bestQ'] > Q_old:
                self.Q_Factor[indexOldState]['bestDecision'] = decision
                self.Q_Factor[indexOldState]['bestQ'] = Q_old

        elif indexOldState != -1 and indexAction == -1 and indexNewState == -1:
            self.n_explored_state += 1
            Q_old = alpha * reward
            self.Q_Factor[indexOldState]['decisionsKnown'].append(
                {
                    'decision': decision,
                    'Q_factor': Q_old
                }
            )
            # update bestQ
            if self.Q_Factor[indexOldState]['bestQ'] > Q_old:
                self.Q_Factor[indexOldState]['bestDecision'] = decision
                self.Q_Factor[indexOldState]['bestQ'] = Q_old
        elif indexOldState != -1 and indexAction == -1 and indexNewState != -1:
            Q_new = self.Q_Factor[indexNewState]['bestQ']
            Q_old = alpha * (reward + gamma * Q_new)
            self.Q_Factor[indexOldState]['decisionsKnown'].append(
                {
                    'decision': decision,
                    'Q_factor': Q_old
                }
            )
            # update bestQ
            if self.Q_Factor[indexOldState]['bestQ'] > Q_old:
                self.Q_Factor[indexOldState]['bestDecision'] = decision
                self.Q_Factor[indexOldState]['bestQ'] = Q_old
        pass

    def toTorchTensor(self, state):
        state_d = state.disposition
        # Definisco stato nella forma che voglio
        state_np = np.zeros(shape=(self.n_item, self.n_rows, self.n_cols))
        for item in range(self.n_item):
            for i in range(0, self.n_rows):
                for j in range(0, self.n_cols):
                    if state_d[i, j] == (item + 1):
                        state_np[item, i, j] = 1
        return torch.from_numpy(state_np).to(torch.float32)

    def learn(self, iterations=10, learning_rate=0.1):
        self.RunSimulation(iterations=iterations)
        # Q(s,a) = V(s^a) Q-value di (s,a) è uguale al valore dello stato post-decisione V(s^a)
        stateDatasetList = []
        target = []
        for dictQ in self.Q_Factor:
            for state_decision in dictQ['decisionsKnown']:
                action = state_decision['decision']
                self.valueFunction.append(state_decision['Q_factor'])
                statePostDecision = copy.copy(dictQ['state'])
                flagMove = True
                for move in action:
                    if statePostDecision.disposition[0, move['col2']] == 0:
                        statePostDecision._move(
                            move['col1'],
                            move['col2']
                        )
                    else:
                        flagMove = False

                statePDTensor = self.toTorchTensor(state=statePostDecision)  # OneHotEncoding
                stateDatasetList.append(statePDTensor)
                if not flagMove:
                    target.append(float(10000))
                else:
                    target.append(state_decision['Q_factor'])

        # Definiamo dataset
        training_data = StatePDDataset(X=stateDatasetList, y=target)
        train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
        # Ora siamo pronti ad allenare la NN
        self.modelNN = self.modelNN.to(torch.float32)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.modelNN.parameters(), lr=learning_rate)
        epochs = 1
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, self.modelNN, loss_fn, optimizer)
        print("Done!")
        pass

    def RunSimulation(self, iterations=10):
        for _ in tqdm(range(iterations)):
            obs = self.env.reset()
            self.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
            state = self.defineState(self.actualDisposition)
            decision = self.explore_decision()
            obs['actual_warehouse'].disposition = copy.deepcopy(self.actualDisposition.disposition)
            for t in range(self.time_limit):
                action = self.get_action(obs=obs)
                obs, reward, info = self.env.step(decision + action)  # passo lista azioni da fare (decision + azioni)
                self.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
                next_state = self.defineState(self.actualDisposition)
                decision = self.explore_decision()
                obs['actual_warehouse'].disposition = copy.deepcopy(self.actualDisposition.disposition)
                self.updateQValuePD(old=state, decision=decision, reward=reward, new=next_state)
                state = copy.deepcopy(next_state)

        print("Simulation has finished!")

    def agentDecisionRandom(self, grid, n_trials=3, n_moves=3):
        # Idea, tra gli stati da valutare è conveniente vedere se è preferibile rimanere nello stato attuale
        action_list = np.zeros((n_trials, 2 * n_moves), dtype='int')
        decision_list = []
        value_list = []
        for k in range(0, n_trials):
            i = 0
            n_rep = 0
            while i < 2 * n_moves:
                col1 = np.random.randint(0, self.n_cols)
                col2 = np.random.randint(0, self.n_cols)
                if self.actualDisposition.disposition[0, col2] == 0 and col1 != col2:
                    action_list[k, i] = col1
                    action_list[k, i + 1] = col2
                    i += 2
                elif n_rep > 10 and self.actualDisposition.disposition[0, col2] != 0:
                    action_list[k, i] = col1
                    action_list[k, i + 1] = col1
                else:
                    n_rep += 1
        for k in range(n_trials):
            i = 0
            flagNoMove = False
            fake_grid = copy.deepcopy(grid)
            while i < n_moves:
                if fake_grid.disposition[0, action_list[k, i + 1]] != 0:
                    flagNoMove = True
                    # value = 100000
                    # value_list.append(value)
                    i = n_moves  # per uscire dal while
                else:
                    fake_grid._move(action_list[k, i], action_list[k, i + 1])
                    i += 2
            if not flagNoMove:
                statePostDecision = self.defineState(fake_grid)
                statePDTensor = self.toTorchTensor(state=statePostDecision)
                # value = self.model(statePDTensor)
                x = self.modelNN(statePDTensor)
                value_list.append(x.item())
            else:
                value_list.append(100000)
        k = np.argmin(np.array(value_list))
        # valutiamo valore stato attuale:
        statePostDecision = self.defineState(self.actualDisposition)
        statePDTensor = self.toTorchTensor(state=statePostDecision)
        value = self.modelNN(statePDTensor)
        valueNoAction = value.detach().numpy()
        if valueNoAction >= value_list[k]:
            best_decision = action_list[k, :]
            j = 0
            while j < 2 * self.n_moves:
                if best_decision[j] != best_decision[j + 1]:
                    decisionDict = {
                        'type': 'M',
                        'col1': best_decision[j],
                        'col2': best_decision[j + 1]
                    }
                    self.actualDisposition._move(
                        col1=best_decision[j],
                        col2=best_decision[j + 1]
                    )
                    decision_list.append(decisionDict)

                j += 2

        else:
            decision_list = []

        return decision_list

    def valueState(self, state):
        statePostDecision = self.defineState(state)
        statePDTensor = self.toTorchTensor(state=statePostDecision)
        # value = self.model(statePDTensor)
        x = torch.flatten(statePDTensor)
        x = F.sigmoid(self.modelNN.fc1(x))
        x = self.modelNN.fc3(x)
        x = x.detach().numpy()
        return x

    def agentDecision(self, grid, probStop=0.1):
        decision_list = []
        working = True
        while working:
            findMin, oneMoveDecision = self.decisionGridSearch(grid)
            if findMin:
                decision_list.append(oneMoveDecision)
                working = findMin and np.random.binomial(1, probStop)
            else:
                working = False
        for move in decision_list:  # aggiorno disposizione che vede agentNN2
            self.actualDisposition._move(
                col1=move['col1'],
                col2=move['col2']
            )
        return decision_list

    def decisionGridSearch(self, grid):
        findMin = False
        possible_col = []
        valueActualState = self.valueState(grid)
        actionMin = np.zeros(3)
        for i in range(self.n_cols):
            possible_col.append(i)
        perm = permutations(possible_col, 2)
        action_list = []
        for (i, j) in list(perm):
            action_list.append([i, j])
        l = len(action_list)
        action_list = np.array(action_list)
        for k in range(l):
            if self.actualDisposition.disposition[0, action_list[k, 1]] == 0:
                fake_grid = copy.deepcopy(grid)
                fake_grid._move(action_list[k, 0], action_list[k, 1])
                value = self.valueState(fake_grid)
                if value < valueActualState:
                    findMin = True
                    actionMin = action_list[k, :]
                    valueActualState = value
        if findMin:
            best_decision = actionMin
            decisionDict = {
                'type': 'M',
                'col1': best_decision[0],
                'col2': best_decision[1]
            }
            self.actualDisposition._move(
                col1=best_decision[0],
                col2=best_decision[1]
            )
            return findMin, decisionDict
        else:
            return findMin, []
