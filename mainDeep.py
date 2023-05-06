import copy
from agent.DeepAgent import DeepAgent
from env.warehouse import Warehouse
from AgentNN import AgentNN
from AgentNN import Net

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
    # agent.learnFrequency(num_episode=1)
    agent.learn(iterations=1)
    # costNoAgent = marshallingWithoutAgent(env, agent)
    # costAgent = marshallingWithAgent(env, agent)

    # L'idea è quella di vedere cosa succede cambiando il numero di parametri
    costNoAgent = []
    costAgent = []
    # for n_cols in [2, 3, 7]:
    #     env = Warehouse(
    #         n_parcel_types=n_parcel_types,
    #         n_rows=n_rows,
    #         n_cols=n_cols
    #     )
    #     obs = env.reset()
    #     # env.plot()
    #     agent = AgentPostDecision(
    #         warehouse=env,
    #         alpha=0.4,
    #         gamma=0.9,
    #         n_item=n_parcel_types,
    #         n_moves=2,  # viene fatta questa ipotesi per gestire al meglio la state-trasformation
    #         time_limit=time_limit,
    #         eps=0.3
    #     )
    #     costNoAgent.append(marshallingWithoutAgent(env, agent))
    #     costAgent.append(marshallingWithAgent(env, agent))

    # print('Finito!')
    # print('Cost no Agent:', costNoAgent)
    # print('Cost with Agent:', costAgent)