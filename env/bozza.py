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
    number_mov = np.random.randint(0, self.n_moves + 1)
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
