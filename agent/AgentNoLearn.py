#!/usr/bin/python3
import numpy as np
from copy import deepcopy


class AgentNoLearn:
    def __init__(self):
        self.actualDisposition = None

    def get_action(self, obs):  # aggiornare sempre cosa vede l'agentNN2!
        self.actualDisposition = deepcopy(obs['actual_warehouse'])
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
        return act

    def learn(self, iterations=10):
        pass
