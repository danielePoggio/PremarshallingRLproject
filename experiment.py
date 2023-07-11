from agent import AgentNN
from agent import AgentNoLearn
from agent import AgentNN2
from env.warehouse import Warehouse
from utils import marshallingWithoutAgent
from utils import marshallingWithAgentNN
from utils import marshallingWithAgentNN2
from utils import plot_3, plot_2d_graph, plot_2


def decide_next_steps(expColumns, expItems, expIterations, expLearningRateRL):
    path = 'C:/Users/39334/Desktop/Poli/PremarshallingRLproject/plot'
    experiment_results = {}
    if expColumns:
        alpha = 0.4
        n_rows = 3
        n_cols = 3
        time_limit = 100
        iterations = 2
        n_parcel_types = 6
        costNoAgent = []
        costAgentNN = []
        costAgentNN2 = []
        timeAgent = []
        numTakenDecision1 = []
        numTakenDecision2 = []
        n_col_list = [3, 4, 5, 6]
        for n_cols in n_col_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = AgentNoLearn()
            agentNN = AgentNN(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                              eps=0.3)
            agentNN2 = AgentNN2(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                                eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agentNN, time_limit, iterations)
            costAgentNN.append(cost)
            numTakenDecision1.append(no_empty_decision)
            cost, time, no_empty_decision = marshallingWithAgentNN2(env, agentNN2, time_limit, iterations)
            costAgentNN2.append(cost)
            numTakenDecision2.append(no_empty_decision)

        plot_3(x_data=n_col_list, yNoLearn=costNoAgent, yNN=costAgentNN, yNN2=costAgentNN2,
               x_label='Number of Columns', y_label='Warehouse Cost', title='col_cost', path=path)
        plot_2(x=n_col_list, y1=numTakenDecision1, y2=numTakenDecision2,x_label='Number of Columns',
               y_label='Number Taken Decisions', title='col_decision', path=path)
        # experiment_results['col'] = {'noAgent': costNoAgent, 'AgentNN': costAgent, 'timeAgent': timeAgent,
        #                              'numDecision': numTakenDecision}

    """ VARIAZIONE NUMERO DI ITEMS """
    if expItems:
        alpha = 0.4
        n_rows = 3
        n_cols = 4
        time_limit = 100
        iterations = 2
        n_parcel_types = 6
        costNoAgent = []
        costAgentNN = []
        costAgentNN2 = []
        numTakenDecision1 = []
        numTakenDecision2 = []
        n_parcel_types_list = [3, 4, 5, 6, 7]
        for n_parcel_types in n_parcel_types_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = AgentNoLearn()
            agentNN = AgentNN(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                              eps=0.3)
            agentNN2 = AgentNN2(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                                eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agentNN, time_limit, iterations)
            costAgentNN.append(cost)
            numTakenDecision1.append(no_empty_decision)
            cost, time, no_empty_decision = marshallingWithAgentNN2(env, agentNN2, time_limit, iterations)
            costAgentNN2.append(cost)
            numTakenDecision2.append(no_empty_decision)

        plot_3(x_data=n_parcel_types_list, yNoLearn=costNoAgent, yNN=costAgentNN, yNN2=costAgentNN2,
               x_label='Number of Parcel Items', y_label='Warehouse Cost', title='parcel_cost', path=path)
        plot_2(x=n_parcel_types_list, y1=numTakenDecision1, y2=numTakenDecision2, x_label='Number of Parcel Items',
               y_label='Number Taken Decisions', title='parcel_decision', path=path)

    """ VARIAZIONE NUMERO DI ITERAZIONI """
    if expIterations:
        alpha = 0.4
        n_rows = 3
        n_cols = 4
        time_limit = 100
        iterations = 2
        n_parcel_types = 6
        costNoAgent = []
        costAgentNN = []
        costAgentNN2 = []
        timeAgent = []
        numTakenDecision1 = []
        numTakenDecision2 = []
        num_iteration_list = list(range(1, 10))
        for iterations in num_iteration_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = AgentNoLearn()
            agentNN = AgentNN(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                              eps=0.3)
            agentNN2 = AgentNN2(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                                eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agentNN, time_limit, iterations)
            costAgentNN.append(cost)
            numTakenDecision1.append(no_empty_decision)
            cost, time, no_empty_decision = marshallingWithAgentNN2(env, agentNN2, time_limit, iterations)
            costAgentNN2.append(cost)
            numTakenDecision2.append(no_empty_decision)

        plot_3(x_data=num_iteration_list, yNoLearn=costNoAgent, yNN=costAgentNN, yNN2=costAgentNN2,
               x_label='Number of Iterations', y_label='Warehouse Cost', title='iterations_cost', path=path)
        plot_2(x=num_iteration_list, y1=numTakenDecision1, y2=numTakenDecision2, x_label='Number of Iterations',
               y_label='Number Taken Decisions', title='iterations_decision', path=path)

    if expLearningRateRL:
        n_rows = 3
        n_cols = 4
        time_limit = 100
        iterations = 2
        n_parcel_types = 6
        costNoAgent = []
        costAgentNN = []
        costAgentNN2 = []
        timeAgent = []
        numTakenDecision1 = []
        numTakenDecision2 = []
        learningRateRL_list = [0.1, 0.3, 0.4, 0.5, 0.6, 0.9, 1]
        for alpha in learningRateRL_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = AgentNoLearn()
            agentNN = AgentNN(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                              eps=0.3)
            agentNN2 = AgentNN2(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit,
                                eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agentNN, time_limit, iterations)
            costAgentNN.append(cost)
            numTakenDecision1.append(no_empty_decision)
            cost, time, no_empty_decision = marshallingWithAgentNN2(env, agentNN2, time_limit, iterations)
            costAgentNN2.append(cost)
            numTakenDecision2.append(no_empty_decision)

        plot_3(x_data=learningRateRL_list, yNoLearn=costNoAgent, yNN=costAgentNN, yNN2=costAgentNN2,
               x_label='Learning Rate Coefficient', y_label='Warehouse Cost', title='LR_cost', path=path)
        plot_2(x=learningRateRL_list, y1=numTakenDecision1, y2=numTakenDecision2, x_label='Learning Rate Coefficient',
               y_label='Number Taken Decisions', title='LR_decision', path=path)

    return experiment_results
