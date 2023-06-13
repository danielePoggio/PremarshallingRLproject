import numpy as np
from tqdm import tqdm
import copy
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from itertools import permutations
from utils import train_loop, compare_np_arr


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
    elem_last_cols1 = state1['needChange']
    freeSpace1 = state1['freeSpace']
    elem_last_cols2 = state2['needChange']
    freeSpace2 = state2['freeSpace']
    flag = compare_np_arr(state1=elem_last_cols1, state2=elem_last_cols2) and compare_np_arr(state1=freeSpace1,
                                                                                             state2=freeSpace2)
    return flag


class AgentEasy:
    def __init__(self, warehouse, alpha, gamma, n_item, time_limit, n_moves=1, eps=0.1, model = 'MLP'):
        self.actualState = None
        self.stateList = []
        self.valueFunction = []
        self.actualDisposition = None
        self.eps = eps
        self.time_limit = time_limit
        self.n_explored_state = 0
        self.n_cols = warehouse.n_cols
        self.n_rows = warehouse.n_rows
        self.n_item = n_item
        self.n_moves = n_moves
        self.n_time = 0
        self.env = copy.deepcopy(warehouse)
        self.actualDecision = None
        self.learningCoeff = alpha
        self.discountCoeff = gamma
        if model == 'linear':
            self.model = linear_model.LinearRegression()
        else:
            self.model = MLPRegressor()
        self.parcelObs = np.zeros(n_item)
        self.parcelFreq = np.zeros(n_item)
        self.tot_parcel = 0
        self.Q_Factor = []  # lista di dizionari tipo
        # {'state' : disposition, 'decisionsKnown' : [], 'bestDecision' : decision,
        # 'bestQ': q}
        # dove la lista in decisionsKnown è composta da dizionari del tipo {'decision': decision, 'Q_factor': Q}
        # SERVE PER RIDURRE TEMPO DI RICERCA COPPIA STATI-AZIONE

    def get_action(self, obs):  # aggiornare sempre cosa vede l'agente!
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

    def defineState(self, disposition):
        needChange = np.zeros(self.n_cols)
        freeSpace = np.zeros(self.n_cols)
        # Osservo la prima riga e la seconda riga di ciascuna colonna
        for j in range(0, self.n_cols):
            if disposition[0, j] == 0:
                freeSpace[j] = 1
            done = False
            i = 0
            while not done:
                if i + 2 >= self.n_rows:
                    done = True
                else:
                    if disposition[i, j] == 0 and disposition[i + 1, j] != 0 and disposition[i + 2, j] != 0:
                        if self.parcelFreq[disposition[i + 1, j] - 1] > self.parcelFreq[disposition[i + 2, j] - 1]:
                            needChange[j] = 1
                        done = True
                        i += 1
                    else:
                        i += 1
        stateDict = {
            'freeSpace': freeSpace,
            'needChange': needChange
        }
        return stateDict

    def explore_decision(self, state):
        choice = np.random.binomial(1, p=self.eps)
        if choice == 1:
            decision = self.exploration(state)
        else:
            decision = self.exploitation(state)
        self.actualDecision = decision
        return decision

    def exploration(self, state):
        state = copy.deepcopy(state)
        decision = []
        n_col = self.n_cols
        number_mov = 1
        n_rep = 0
        n_action = 0
        while n_action < number_mov:
            n_rep += 1
            if n_rep > 1000:
                print("HELP")
            col1 = np.random.randint(0, n_col)
            col2 = np.random.randint(0, n_col)
            if state['freeSpace'][col2] == 1 and col1 != col2:
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

    def exploitation(self, state):
        # allora devo valutare il minimo tra i Q-Factor
        findState = 0
        q_factor = self.Q_Factor
        for dictQFactor in q_factor:
            # guardare bene l'if perchè tratta vettori di lunghezza diversa!!
            if compareState(dictQFactor['state'],
                            state):  # ho trovato lo stato
                findState = 1
                changeDisposition = dictQFactor['bestDecision']  # selezione l'azione migliore che conosco
                for action in changeDisposition:
                    col1 = action['col1']
                    col2 = action['col2']

                    self.actualDisposition._move(
                        col1,
                        col2
                    )
        if findState == 0:  # non ho trovato lo stato
            changeDisposition = self.exploration(state)

        return changeDisposition

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

    def forRegression(self, state):
        # state_np = np.zeros(4 * self.n_cols)
        # state_np[0:self.n_cols] = state['freeSpace']
        # state_np[self.n_cols:2 * self.n_cols] = state['needChange']
        # state_np[2 * self.n_cols:3 * self.n_cols] = state['action'][0, :]
        # state_np[3 * self.n_cols:4 * self.n_cols] = state['action'][1, :]
        return state['freeSpace'].tolist() + state['needChange'].tolist() + state['action'][0, :].tolist() + \
            state['action'][1, :].tolist()

    def learn(self, iterations=10, learning_rate=0.1):
        self.RunSimulation(iterations=iterations)
        stateDatasetList = []
        target = []
        for dictQ in self.Q_Factor:
            for state_decision in dictQ['decisionsKnown']:
                action = state_decision['decision']
                self.valueFunction.append(state_decision['Q_factor'])
                state_action = copy.copy(dictQ['state'])  # copio lo stato
                for move in action:
                    if state_action['freeSpace'][move['col2']] == 1:
                        action_list = np.zeros((2, self.n_cols))
                        action_list[0, move['col1']] = 1
                        action_list[1, move['col2']] = 1
                        state_action['action'] = action_list
                        target.append(float(state_decision['Q_factor']))
                    else:
                        action_list = np.zeros((2, self.n_cols))
                        action_list[move['col1']] = 1
                        action_list[move['col2']] = 1
                        state_action['action'] = action_list
                        target.append(float(10000))

                statePDTensor = self.forRegression(state=state_action)
                stateDatasetList.append(statePDTensor)

        # Ora siamo pronti ad allenare la NN
        X = np.array(stateDatasetList)
        y = np.array(target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        self.model.fit(X_train, y_train)
        print('R^2:', self.model.score(X_test, y_test))

    def RunSimulation(self, iterations=10):
        for _ in tqdm(range(iterations)):
            obs = self.env.reset()  # faccio resettare da capo l'ambiente
            for t in range(self.time_limit):
                self.actualDisposition = copy.deepcopy(obs['actual_warehouse'])  # Agente vedo stato
                state = self.defineState(self.actualDisposition.disposition)  # Agente estrapola info che gli servono
                decision = self.explore_decision(state)
                obs['actual_warehouse'].disposition = copy.deepcopy(self.actualDisposition.disposition)
                actOrder = self.get_action(obs)  # g2 nella pipeline
                action = decision + actOrder
                obs, reward, info = self.env.step(action)
                whatsee = compareDisposition(self.actualDisposition.disposition, obs['actual_warehouse'].disposition)
                if not whatsee:
                    print('Alert!')
                    print(self.actualDisposition.disposition == obs['actual_warehouse'].disposition)
                next_state = self.defineState(obs['actual_warehouse'].disposition)
                self.updateQValuePD(old=state, decision=decision, reward=reward, new=next_state)

        print("Simulation has finished!")

    def valueState(self, state, action):
        state_action = self.defineState(state)
        state_action['action'] = action
        statePDTensor = self.forRegression(state=state_action)
        statePDTensor = np.array(statePDTensor).reshape(1, -1)
        value = self.model.predict(statePDTensor)
        return value

    def agentDecision(self, grid, probStop=0.1):
        decision_list = []
        working = True
        while working:
            findMin, oneMoveDecision, grid = self.decisionGridSearch(grid)
            if findMin:
                if oneMoveDecision != [] and grid.disposition[self.n_rows - 1, oneMoveDecision['col1']] != 0 \
                        and grid.disposition[0, oneMoveDecision['col2']] == 0:
                    grid._move(
                        oneMoveDecision['col1'],
                        oneMoveDecision['col2']
                    )
                    decision_list.append(oneMoveDecision)
                working = findMin and np.random.binomial(1, probStop)
            else:
                working = False

        for move in decision_list:  # aggiorno stato agente
            self.actualDisposition._move(
                col1=move['col1'],
                col2=move['col2']
            )
        return decision_list

    def decisionGridSearch(self, grid):
        toll = 1.0e-02
        findMin = False
        possible_col = []
        actionMin = np.zeros((2, self.n_cols))
        valueActualState = self.valueState(grid.disposition, actionMin)
        valueActualState = valueActualState.item()
        for i in range(self.n_cols):
            possible_col.append(i)
        perm = permutations(possible_col, 2)
        action_list = []
        for (i, j) in list(perm):
            action_list.append([i, j])
        l = len(action_list)
        action_list = np.array(action_list)
        kBest = -1
        for k in range(l):
            action = np.zeros((2, self.n_cols))
            noAction = True
            if grid.disposition[self.n_rows - 1, action_list[k, 0]] != 0:
                action[0, action_list[k, 0]] = 1
                action[1, action_list[k, 1]] = 1
                noAction = False
            value = self.valueState(grid.disposition, action)
            value = value.item()
            if value <= valueActualState + toll and noAction is False:
                actionMin = action
                valueActualState = value
                findMin = True
                kBest = k
        if findMin:
            best_decision = action_list[kBest, :]
            if grid.disposition[self.n_rows - 1, best_decision[0]] != 0:
                decisionDict = {
                    'type': 'M',
                    'col1': best_decision[0],
                    'col2': best_decision[1]
                }
                return findMin, decisionDict, grid
            else:
                return findMin, [], grid

        else:
            return findMin, [], grid
