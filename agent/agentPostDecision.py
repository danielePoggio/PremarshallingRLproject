import numpy as np
from tqdm import tqdm
import copy
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from itertools import permutations


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
    elem_last_cols1 = state1['elem_last_cols']
    freeSpace1 = state1['freeSpace']
    elem_last_cols2 = state2['elem_last_cols']
    freeSpace2 = state2['freeSpace']
    flag = compare_np_arr(state1=elem_last_cols1, state2=elem_last_cols2) and compare_np_arr(state1=freeSpace1,
                                                                                             state2=freeSpace2)
    return flag


class AgentPostDecision:
    def __init__(self, warehouse, alpha, gamma, n_item, time_limit, n_moves=2, eps=0.1):
        self.coeff = None
        self.intercept = None
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
        freeSpace = np.zeros(self.n_cols)
        numObjCol = np.zeros(self.n_cols)
        # Osservo la prima riga e la seconda riga di ciascuna colonna
        two_elem_col = np.zeros((2, self.n_cols))
        for j in range(0, self.n_cols):
            objCol = 0
            if disposition[0, j] == 0:
                freeSpace[j] = 1
            for i in range(self.n_rows):
                if disposition[i, j] != 0:
                    objCol += 1
            numObjCol[j] = objCol
            done = False
            i = 0
            item_found = 0
            while not done:
                if disposition[i, j] != 0:
                    two_elem_col[item_found, j] = self.parcelFreq[disposition[i, j] - 1]
                    item_found += 1
                i += 1
                if item_found == 2 or i == self.n_rows:
                    done = True
        for j in range(self.n_cols):  # check
            if two_elem_col[1, j] == 0:
                two_elem_col[1, j] = two_elem_col[0, j]
                two_elem_col[0, j] = 0

        stateDict = {
            'freeSpace': numObjCol,
            'elem_last_cols': two_elem_col
        }
        return stateDict

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

    def stateRegression(self, state):
        # return np.reshape(state, 2*self.n_cols).tolist()
        return state['freeSpace'].tolist() + np.reshape(state['elem_last_cols'], 2 * self.n_cols).tolist()

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
                self.updateQValuePD(old=state, decision=decision, reward=reward, new=next_state)

        # IDEA CHIAVE: Q(s,a) = V(s^a) cioè il post-decision state!
        for dictQ in self.Q_Factor:
            for state_decision in dictQ['decisionsKnown']:
                action = state_decision['decision']
                self.valueFunction.append(state_decision['Q_factor'])
                statePostDecision = copy.copy(dictQ['state']['elem_last_cols'])
                for move in action:
                    col1 = move['col1']
                    col2 = move['col2']
                    if dictQ['state']['freeSpace'][col2] < self.n_rows:  # posso fare spostamento
                        if statePostDecision[0, col1] != 0:  # esiste oggetto da spostare
                            temp = statePostDecision[0, col1]
                            statePostDecision[0, col1] = 0
                            if statePostDecision[0, col2] != 0:
                                statePostDecision[1, col2] = statePostDecision[0, col2]
                                statePostDecision[0, col2] = temp
                                dictQ['state']['freeSpace'][col1] -= 1
                                dictQ['state']['freeSpace'][col2] += 1
                            elif statePostDecision[0, col2] == 0 and statePostDecision[1, col2] != 0:
                                statePostDecision[0, col2] = temp
                                dictQ['state']['freeSpace'][col1] -= 1
                                dictQ['state']['freeSpace'][col2] += 1
                            else:
                                statePostDecision[1, col2] = temp
                                dictQ['state']['freeSpace'][col1] -= 1
                                dictQ['state']['freeSpace'][col2] += 1
                        elif statePostDecision[0, col1] == 0 and statePostDecision[1, col1] != 0:
                            temp = statePostDecision[1, col1]
                            statePostDecision[1, col1] = 0
                            if statePostDecision[0, col2] != 0:
                                statePostDecision[1, col2] = statePostDecision[0, col2]
                                statePostDecision[0, col2] = temp
                                dictQ['state']['freeSpace'][col1] -= 1
                                dictQ['state']['freeSpace'][col2] += 1
                            elif statePostDecision[0, col2] == 0 and statePostDecision[1, col2] != 0:
                                statePostDecision[0, col2] = temp
                                dictQ['state']['freeSpace'][col1] -= 1
                                dictQ['state']['freeSpace'][col2] += 1
                            else:
                                statePostDecision[1, col2] = temp
                                dictQ['state']['freeSpace'][col1] -= 1
                                dictQ['state']['freeSpace'][col2] += 1
                        elif statePostDecision[0, col1] == 0 and statePostDecision[1, col1] == 0:
                            print('Non si può fare nulla!')

                    else:
                        print('Non si può fare nulla (pt. 2)')

                stateForReg = {
                    'freeSpace': dictQ['state']['freeSpace'],
                    'elem_last_cols': statePostDecision
                }
                self.stateList.append(self.stateRegression(stateForReg))

        X = np.array(self.stateList)
        y = np.array(self.valueFunction)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        lin_model = linear_model.LinearRegression()
        modelNN = MLPRegressor()
        lin_model.fit(X_train, y_train)
        modelNN.fit(X_train, y_train)
        print('R^2 of NN:', modelNN.score(X_test, y_test))
        print('R^2 of Linear Model:', lin_model.score(X_test, y_test))
        self.modelNN = modelNN
        self.lin_model = lin_model
        self.coeff = lin_model.coef_
        self.intercept = lin_model.intercept_

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
        number_mov = np.random.randint(0, self.n_moves+1)
        n_rep = 0
        n_action = 0
        while n_action < number_mov:
            n_rep += 1
            if n_rep > 1000:
                print("HELP")
            col1 = np.random.randint(0, n_col)
            col2 = np.random.randint(0, n_col)
            if state['freeSpace'][col2] < self.n_rows and col1 != col2:
                # eseguo azione se le colonne non sono identiche e se la colonna 2 non è piena e se la colonna 1 non
                # è vuota
                state['freeSpace'][col1] -= 1
                state['freeSpace'][col2] += 1
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

    def agentDecision(self, grid):
        possible_col = []
        for i in range(self.n_cols):
            possible_col.append(i)
        perm = permutations(possible_col, 2)
        action_list = []
        for (i, j) in list(perm):
            action_list.append([i, j])
        l = len(action_list)
        action_list = np.array(action_list)
        value_list = []
        for k in range(l):
            if self.actualDisposition.disposition[0, action_list[k, 1]] != 0:
                value = 100000
                value_list.append(value)
            else:
                fake_grid = copy.deepcopy(grid)
                fake_grid._move(action_list[k, 0], action_list[k, 1])
                statePostDecision = self.defineState(fake_grid.disposition)
                stateReg = np.array(self.stateRegression(statePostDecision))
                value = self.intercept + np.dot(stateReg, self.coeff)
                value_list.append(value)
        k = np.argmin(np.array(value_list))
        best_decision = action_list[k, :]
        decisionDict = {
            'type': 'M',
            'col1': best_decision[0],
            'col2': best_decision[1]
        }
        self.actualDisposition._move(
            col1=best_decision[0],
            col2=best_decision[1]
        )
        return [decisionDict]

    def agentDecisionRandom(self, grid, n_trials=3):
        action_list = np.zeros((n_trials, 2*self.n_moves), dtype='int')
        decision_list = []
        value_list = []
        for k in range(0, n_trials):
            i = 0
            n_rep = 0
            while i < self.n_moves:
                col1 = np.random.randint(0, self.n_cols)
                col2 = np.random.randint(0, self.n_cols)
                if self.actualDisposition.disposition[0, col2] == 0:
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
            fake_grid = copy.deepcopy(grid)
            while i < self.n_moves:
                if self.actualDisposition.disposition[0, action_list[k, i+1]] != 0:
                    value = 100000
                    value_list.append(value)
                    i = self.n_moves  # per uscire dal while
                else:
                    fake_grid._move(action_list[k, i], action_list[k, i+1])
                    i += 2
            statePostDecision = self.defineState(fake_grid.disposition)
            stateReg = np.array(self.stateRegression(statePostDecision))
            value = self.intercept + np.dot(stateReg, self.coeff)
            value_list.append(value)
        k = np.argmin(np.array(value_list))
        best_decision = action_list[k, :]
        j = 0
        while j < 2*self.n_moves:
            if best_decision[j] != best_decision[j+1]:
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

        return decision_list

    # def agentDecisionRandom(self, obs, n_trials=3):
    #     state = copy.deepcopy(self.actualDisposition)
    #     # genero possibili azioni
    #     action_list = np.zeros((n_trials, 2 * self.n_moves), dtype='int')
    #     value_list = []
    #     for k in range(0, n_trials):
    #         i = 0
    #         while i < self.n_moves:
    #             col1 = np.random.randint(0, self.n_cols)
    #             col2 = np.random.randint(0, self.n_cols)
    #             action_list[k, i] = col1
    #             action_list[k, i + 1] = col2
    #             i += 2
    #     for k in range(n_trials):
    #         state_fake = copy.deepcopy(state)
    #         f


