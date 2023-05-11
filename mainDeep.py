import copy

from env.warehouse import Warehouse
from agent.AgentNN2 import AgentNN2 as AgentNN



def marshallingWithoutAgent(enviroment, agente):
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


def marshallingWithAgent(enviroment, agente):
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


if __name__ == '__main__':
    # parameters
    n_rows = 3
    n_cols = 3
    time_limit = 100
    n_parcel_types = 5

    env = Warehouse(
        n_parcel_types=n_parcel_types,
        n_rows=n_rows,
        n_cols=n_cols
    )
    obs = env.reset()
    # env.plot()
    agent = AgentNN(env, alpha=0.6, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit)
    agent.learnFrequency(num_episode=1)
    agent.learn(iterations=1)
    obs = env.reset()
    agent.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
    tot_cost = 0
    # Partiamo con le iterazioni
    for t in range(time_limit):
        print('Order:', obs['order'])
        print('New Parcel:', obs['new_parcel'])
        decision = agent.agentDecision(grid=agent.actualDisposition, probStop=0.0010)  # prende decisione
        print('Decision:', decision)
        action = agent.get_action(obs=obs)  # risolve ordini
        print(action)
        obs, cost, info = env.step(decision + action)
        agent.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
        tot_cost += cost
        print(env.disposition.disposition)
        # env.plot()
        print("---")
    print(tot_cost)
    costNoAgent = []
    costAgent = []
    # for n_cols in [3]:
    #     env = Warehouse(
    #         n_parcel_types=n_parcel_types,
    #         n_rows=n_rows,
    #         n_cols=n_cols
    #     )
    #     obs = env.reset()
    #     # env.plot()
    #     agent = AgentNN(
    #         warehouse=env,
    #         alpha=0.4,
    #         gamma=0.9,
    #         n_item=n_parcel_types,
    #         n_moves=1,  # viene fatta questa ipotesi per gestire al meglio la state-trasformation
    #         time_limit=time_limit,
    #         eps=0.3
    #     )
    #     # costNoAgent.append(marshallingWithoutAgent(env, agent))
    #     costAgent.append(marshallingWithAgent(env, agent))

    print('Finito!')
    print('Cost no Agent:', costNoAgent)
    print('Cost with Agent:', costAgent)
