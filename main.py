#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
from agent.agentPostDecision import AgentPostDecision
from env.warehouse import Warehouse


def marshallingWithoutAgent(enviroment, agente):
    obs = enviroment.reset()
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])  # azione necessaria percè l'agente veda
    # l'ambiente
    tot_cost = 0
    for t in range(time_limit):
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        decision = []  # siccome l'agente non prende decisioni è come se la lista delle decisioni fosse vuota
        action = agente.get_actionPostDecision(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = enviroment.step(decision + action)
        agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
        print(enviroment.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    return tot_cost


def marshallingWithAgent(enviroment, agente):
    # Eseguiamo apprendimento dell'agente
    agente.learnFrequency(num_episode=10)
    agente.learn(iterations=10)
    # resettiamo ambiente
    obs = enviroment.reset()
    tot_cost = 0
    # Partiamo con le iterazioni
    for t in range(time_limit):
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        decision = agente.agentDecisionRandom(grid=agente.actualDisposition, n_trials=2)  # prende decisione
        action = agente.get_actionPostDecision(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = enviroment.step(decision + action)
        agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
        print(enviroment.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    return tot_cost


# np.random.seed(1)

if __name__ == '__main__':
    # parameters
    n_rows = 3
    n_cols = 3
    time_limit = 100
    n_parcel_types = 5
    costNoAgent = []
    costAgent = []
    for n_cols in [4]:
        env = Warehouse(
            n_parcel_types=n_parcel_types,
            n_rows=n_rows,
            n_cols=n_cols
        )
        obs = env.reset()
        # env.plot()
        agent = AgentPostDecision(
            warehouse=env,
            alpha=0.4,
            gamma=0.9,
            n_item=n_parcel_types,
            n_moves=2,  # viene fatta questa ipotesi per gestire al meglio la state-trasformation
            time_limit=time_limit,
            eps=0.3
        )
        costNoAgent.append(marshallingWithoutAgent(env, agent))
        costAgent.append(marshallingWithAgent(env, agent))

    print('Finito!')
    print('Cost no Agent:', costNoAgent)
    print('Cost with Agent:', costAgent)
