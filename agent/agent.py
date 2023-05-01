#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


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


def dictStateEquals(state1, state2):
    higherObject1 = state1['higherObject']
    numObjCol1 = state1['numObjCol']
    higherObject2 = state2['higherObject']
    numObjCol2 = state2['numObjCol']
    control1 = np.sum(np.where(higherObject1 == higherObject2, 1, 0))
    control2 = np.sum(np.where(numObjCol1 == numObjCol2, 1, 0))
    flag = (control1 + control2) / (2 * len(higherObject1))
    if flag < 1:
        flag = 0
    else:
        flag = int(flag)
    return flag


class Agent:
    def __init__(self, warehouse, alpha, gamma, n_item, time_limit, n_moves=2, eps=0.1):
        self.modelNN = None
        self.lin_model = None
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
        self.num_actions = (n_item + 1) ** (2 * n_moves)
        self.deliveryObj = np.zeros(n_item + 1)
        self.deliveryProb = np.zeros(n_item + 1)
        self.parcelObs = np.zeros(n_item + 1)
        self.parcelProb = np.zeros(n_item + 1)
        self.tot_order = 0
        self.tot_parcel = 0
        self.Q_Factor = []  # lista di dizionari tipo
        # {'state' : disposition, 'decisionsKnown' : [], 'bestDecision' : decision,
        # 'bestQ': q}
        # dove la lista in decisionsKnown è composta da dizionari del tipo {'decision': decision, 'Q_factor': Q}
        # SERVE PER RIDURRE TEMPO DI RICERCA COPPIA STATI-AZIONE

    def defineState(self, state):
        freeSpace = np.zeros(self.n_cols)
        needChange = np.zeros(self.n_cols)
        numObjCol = np.zeros(self.n_cols)
        # definisco le frequenze dei vari oggetti all'interno del magazzino
        frequency = np.zeros(self.n_item)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if state[i, j] != 0:
                    frequency[state[i, j] - 1] += 1

        # Osservo la prima riga e la seconda riga di ciascuna colonna
        two_elem_col = np.zeros((2, self.n_cols), dtype='int32')
        for j in range(0, self.n_cols):
            objCol = 0
            if state[0, j] == 0:
                freeSpace[j] = 1
            for i in range(self.n_rows):
                if state[i, j] != 0:
                    objCol += 1
            numObjCol[j] = objCol
            done = False
            i = 0
            item_found = 0
            while not done:
                if state[i, j] != 0:
                    two_elem_col[item_found, j] = state[i, j]
                    item_found += 1
                i += 1
                if item_found == 2 or i == self.n_rows:
                    done = True

        # valuto se l'oggetto nella prima riga ha una frequenza maggiore di quello nella seconda
        for j in range(self.n_cols):
            if frequency[two_elem_col[0, j] - 1] != 0 and frequency[two_elem_col[1, j] - 1] != 0:
                if frequency[two_elem_col[0, j] - 1] > frequency[two_elem_col[1, j] - 1]:
                    needChange[j] = 1

        stateDict = {
            'freeSpace': numObjCol,
            'needChange': needChange
        }
        return stateDict

    def stateRegression(self, stateDict):
        state = stateDict['freeSpace'].tolist() + stateDict['needChange'].tolist()
        return state

    def get_action(self, obs):  # aggiornare sempre cosa vede l'agente!
        self.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
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

    def explore_decision(self, state):
        choice = np.random.binomial(1, p=self.eps)
        if choice == 1:
            decision = self.exploration(state)
        else:
            decision = self.exploitation(state)
        self.actualDecision = decision
        return decision

    def exploration(self, magazzino):
        decision = []
        n_col = self.n_cols
        number_mov = self.n_moves
        n_rep = 0
        n_action = 0
        while n_action < number_mov:
            n_rep += 1
            if n_rep > 1000:
                print("HELP")
            col1 = np.random.randint(0, n_col)
            col2 = np.random.randint(0, n_col)
            if magazzino['freeSpace'][col2] < self.n_rows and col1 != col2:
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
            if self.compareDisposition(dictQFactor['state'],
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

    def updateQValue(self, old, decision, reward, new):  # aggiorno Q_Factor
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
        if self.compareDisposition(old, new):
            equalState = 1
        for q in q_factor:
            if self.compareDisposition(q['state'], old):  # lo stato era già stato visitato
                indexOldState = j
                i = 0
                for act in q['decisionsKnown']:
                    if act['decision'] == decision:  # la coppia stato-azione era già stata visitata-> allora esiste un
                        # Q-factor
                        indexAction = i
                    i = i + 1
                if equalState == 1:  # non ha senso andare avanti con il ciclo for
                    break
            if self.compareDisposition(q['state'], new):  # stato già visitato
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

    def learn(self, iterations=10):
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
                self.updateQValue(old=state, decision=decision, reward=reward, new=next_state)
        # IDEA CHIAVE: Q(s,a) = V(s^a) cioè il post-decision state!
        n_explored_state = len(self.Q_Factor)
        i = 0
        for dictQ in self.Q_Factor:
            for state_decision in dictQ['decisionsKnown']:
                action = state_decision['decision']
                self.valueFunction.append(state_decision['Q_factor'])
                statePostDecision = copy.copy(dictQ['state'])
                for move in action:
                    statePostDecision._move(
                        move['col1'],
                        move['col2']
                    )
                self.stateList.append(self.stateRegression(stateDict=state_decision))

        X = np.array(self.stateList, dtype='int32')
        y = np.array(self.valueFunction)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        lin_model = linear_model.LinearRegression()
        modelNN = MLPRegressor()
        lin_model.fit(X_train, y_train)
        modelNN.fit(X_train, y_train)
        print('R^2 of NN:', modelNN.score(X_test, y_test))
        print('R^2 of Linear Model:', lin_model.score(X_test, y_test))
        self.modelNN = modelNN
        self.lin_model = lin_model

    def compareDisposition(self, a, b):
        aFreeSpace = a['freeSpace']
        bFreeSpace = b['freeSpace']
        aneedChange = a['needChange']
        bneedChange = b['needChange']
        flag = np.array_equal(aFreeSpace, bFreeSpace) and np.array_equal(aneedChange, bneedChange)
        return flag

    def learnPostDecision(self, iterations=10):
        for _ in tqdm(range(iterations)):
            obs = self.env.reset()  # faccio resettare da capo l'ambiente
            for t in range(self.time_limit):
                self.actualDisposition = copy.deepcopy(obs['actual_warehouse'])  # Agente vedo stato
                state = copy.deepcopy(self.actualDisposition.disposition)  # Agente estrapola info che gli servono
                decision = self.explore_decisionPD(state)
                obs['actual_warehouse'].disposition = copy.deepcopy(self.actualDisposition.disposition)
                actOrder = self.get_action(obs)  # g2 nella pipeline
                action = decision + actOrder
                obs, reward, info = self.env.step(action)
                whatsee = compareDisposition(self.actualDisposition.disposition, obs['actual_warehouse'].disposition)
                if not whatsee:
                    print('Alert!')
                    print(self.actualDisposition.disposition == obs['actual_warehouse'].disposition)
                next_state = copy.deepcopy(obs['actual_warehouse'].disposition)
                self.updateQValuePD(old=state, decision=decision, reward=reward, new=next_state)
        # IDEA CHIAVE: Q(s,a) = V(s^a) cioè il post-decision state!
        n_explored_state = len(self.Q_Factor)
        i = 0
        for dictQ in self.Q_Factor:
            for state_decision in dictQ['decisionsKnown']:
                action = state_decision['decision']
                self.valueFunction.append(state_decision['Q_factor'])
                statePostDecision = copy.copy(dictQ['state'])
                for move in action:
                    statePostDecision._move(
                        move['col1'],
                        move['col2']
                    )
                self.stateList.append(self.stateRegression(stateDict=state_decision))

        X = np.array(self.stateList, dtype='int32')
        y = np.array(self.valueFunction)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        lin_model = linear_model.LinearRegression()
        modelNN = MLPRegressor()
        lin_model.fit(X_train, y_train)
        modelNN.fit(X_train, y_train)
        print('R^2 of NN:', modelNN.score(X_test, y_test))
        print('R^2 of Linear Model:', lin_model.score(X_test, y_test))
        self.modelNN = modelNN
        self.lin_model = lin_model

    def explore_decisionPD(self, state):
        choice = np.random.binomial(1, p=self.eps)
        if choice == 1:
            decision = self.explorationPD(state)
        else:
            decision = self.exploitationPD(state)
        self.actualDecision = decision
        return decision

    def explorationPD(self, state):
        decision = []
        n_col = self.n_cols
        number_mov = self.n_moves
        n_rep = 0
        n_action = 0
        while n_action < number_mov:
            n_rep += 1
            if n_rep > 1000:
                print("HELP")
            col1 = np.random.randint(0, n_col)
            col2 = np.random.randint(0, n_col)
            if state[0, col2] == 0 and col1 != col2:
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

    def exploitationPD(self, state):
        # allora devo valutare il minimo tra i Q-Factor
        findState = 0
        q_factor = self.Q_Factor
        for dictQFactor in q_factor:
            # guardare bene l'if perchè tratta vettori di lunghezza diversa!!
            if compareDisposition(dictQFactor['state'],
                                  state):  # ho trovato lo stato
                findState = 1
                decision = dictQFactor['bestDecision']  # selezione l'azione migliore che conosco
                for action in decision:
                    col1 = action['col1']
                    col2 = action['col2']

                    self.actualDisposition._move(
                        col1,
                        col2
                    )
        if findState == 0:  # non ho trovato lo stato
            decision = self.explorationPD(state)

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
        if compareDisposition(old, new):
            equalState = 1
        for q in q_factor:
            if compareDisposition(q['state'], old):  # lo stato era già stato visitato
                indexOldState = j
                i = 0
                for act in q['decisionsKnown']:
                    if act['decision'] == decision:  # la coppia stato-azione era già stata visitata-> allora esiste un
                        # Q-factor
                        indexAction = i
                    i = i + 1
                if equalState == 1:  # non ha senso andare avanti con il ciclo for
                    break
            if compareDisposition(q['state'], new):  # stato già visitato
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

