import copy
from env.warehouse import Warehouse
from agent import AgentEasy as Agent
import time

alpha = 0.9
n_rows = 3
n_cols = 3
time_limit = 100
iterations = 2
n_parcel_types = 5
gamma = 0.9
env = Warehouse(
    n_parcel_types=n_parcel_types,
    n_rows=n_rows,
    n_cols=n_cols
)
obs = env.reset()
agente = Agent(env, alpha, gamma, n_parcel_types, time_limit, n_moves=2, eps=0.1)
start_time = time.time()
agente.learnFrequency(num_episode=1)
# Eseguiamo apprendimento dell'agente
agente.learn(iterations=iterations)
# Resettiamo ambiente
obs = env.reset()
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
    obs, cost, info = env.step(decision + action)
    # Agente osserva nuova disposizione per prendere decisione futura
    agente.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
    tot_cost += cost
print(tot_cost)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
# if __name__ == '__main__':
#     # parameters
#     n_rows = 3
#     n_cols = 3
#     time_limit = 100
#     n_parcel_types = 5
#
#     env = Warehouse(
#         n_parcel_types=n_parcel_types,
#         n_rows=n_rows,
#         n_cols=n_cols
#     )
#     obs = env.reset()
#     # env.plot()
#     agent = AgentNN(env, alpha=0.6, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit)
#     agent.learnFrequency(num_episode=1)
#     agent.learn(iterations=10)
#     obs = env.reset()
#     agent.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
#     tot_cost = 0
#     # Partiamo con le iterazioni
#     for t in range(time_limit):
#         print('Order:', obs['order'])
#         print('New Parcel:', obs['new_parcel'])
#         decision = agent.agentDecision(grid=agent.actualDisposition, probStop=0.0010)  # prende decisione
#         print('Decision:', decision)
#         action = agent.get_action(obs=obs)  # risolve ordini
#         print(action)
#         obs, cost, info = env.step(decision + action)
#         agent.actualDisposition = copy.deepcopy(obs['actual_warehouse'])
#         tot_cost += cost
#         print(env.disposition.disposition)
#         # env.plot()
#         print("---")
#     print(tot_cost)
#     costNoAgent = []
#     costAgent = []
#
#     print('Finito!')
#     print('Cost no Agent:', costNoAgent)
#     print('Cost with Agent:', costAgent)
