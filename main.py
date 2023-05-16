#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
from agent import AgentNN as Agent
from env.warehouse import Warehouse
from utils import *


if __name__ == '__main__':
    # parameters
    n_rows = 3
    # n_cols = 3
    time_limit = 100
    iterations = 2
    n_parcel_types = 5
    costNoAgent = []
    costAgent = []
    timeAgent = []
    for n_cols in [2, 3, 4, 5]:
        env = Warehouse(
            n_parcel_types=n_parcel_types,
            n_rows=n_rows,
            n_cols=n_cols
        )
        obs = env.reset()
        agent = Agent(warehouse=env, alpha=0.4, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit, eps=0.3)
        costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
        cost, time = marshallingWithAgent(env, agent, time_limit, iterations)
        costAgent.append(cost)
        timeAgent.append(time)

    plot_comparison(x_data=[2, 3, 4, 5] , y1_data=costAgent, y2_data=costNoAgent, x_label='Numero colonne Warehouse',
                    y_label='Costo magazzino', title='', label1='Costo con agente', label2='costo senza agente')

    # plot_2d_graph(x_data=[2, 3, 4, 5], y_data=costAgent, x_label='Numero colonne Warehouse', y_label='Costo magazzino',
    #               title='Grafico costo del magazzino al variare del numero di colonne')
    # plot_2d_graph(x_data=[2, 3, 4, 5], y_data=timeAgent, x_label='Numero colonne Warehouse', y_label='Tempo',
    #               title='Grafico tempo di esecuzione al variare del numero di colonne')

    print('Finito!')
    print('Cost no Agent:', costNoAgent)
    print('Cost with Agent:', costAgent)
