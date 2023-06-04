from agent import AgentNN2 as Agent
from env.warehouse import Warehouse
from utils import marshallingWithoutAgent
from utils import marshallingWithAgentNN2 as marshallingWithAgentNN
from utils import plot_comparison, plot_2d_graph


def decide_next_steps(expColumns, expItems, expIterations, expLearningRateRL):
    experiment_results = {}
    if expColumns:
        alpha = 0.9
        n_rows = 3
        n_cols = 3
        time_limit = 100
        iterations = 2
        n_parcel_types = 5
        costNoAgent = []
        costAgent = []
        timeAgent = []
        numTakenDecision = []
        n_col_list = [3, 4, 5, 6]
        for n_cols in n_col_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = Agent(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit, eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agent, time_limit, iterations)
            costAgent.append(cost)
            timeAgent.append(time)
            numTakenDecision.append(no_empty_decision)

        plot_comparison(x_data=n_col_list, y1_data=costAgent, y2_data=costNoAgent, x_label='Numero colonne Warehouse',
                        y_label='Costo magazzino', title='', label1='Costo con agente', label2='costo senza agente')
        plot_2d_graph(n_col_list, timeAgent, 'Numero colonne', 'Tempo esecuzione', '')
        plot_2d_graph(n_col_list, numTakenDecision, 'Numero colonne', 'Numero di decisioni prese', '')
        experiment_results['col'] = {'noAgent': costNoAgent, 'Agent': costAgent, 'timeAgent': timeAgent,
                                     'numDecision': numTakenDecision}

    """ VARIAZIONE NUMERO DI ITEMS """
    if expItems:
        alpha = 0.9
        n_rows = 3
        n_cols = 3
        time_limit = 100
        iterations = 2
        n_parcel_types = 5
        costNoAgent = []
        costAgent = []
        timeAgent = []
        numTakenDecision = []
        n_parcel_types_list = [3, 4, 5, 6, 7]
        for n_parcel_types in n_parcel_types_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = Agent(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit, eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agent, time_limit, iterations)
            costAgent.append(cost)
            timeAgent.append(time)
            numTakenDecision.append(no_empty_decision)

        plot_comparison(x_data=n_parcel_types_list, y1_data=costAgent, y2_data=costNoAgent,
                        x_label='Numero items differenti',
                        y_label='Costo magazzino', title='', label1='Costo con agente', label2='costo senza agente')
        plot_2d_graph(n_parcel_types_list, timeAgent, 'Numero items differenti', 'Tempo esecuzione', '')
        plot_2d_graph(n_parcel_types_list, numTakenDecision, 'Numero items differenti', 'Numero di decisioni prese', '')
        experiment_results['items'] = {'noAgent': costNoAgent, 'Agent': costAgent, 'timeAgent': timeAgent,
                                       'numDecision': numTakenDecision}

    """ VARIAZIONE NUMERO DI ITERAZIONI """
    if expIterations:
        alpha = 0.9
        n_rows = 3
        n_cols = 3
        time_limit = 100
        iterations = 2
        n_parcel_types = 5
        costNoAgent = []
        costAgent = []
        timeAgent = []
        numTakenDecision = []
        num_iteration_list = list(range(1, 10))
        for iterations in num_iteration_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = Agent(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit, eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agent, time_limit, iterations)
            costAgent.append(cost)
            timeAgent.append(time)
            numTakenDecision.append(no_empty_decision)

        plot_comparison(x_data=num_iteration_list, y1_data=costAgent, y2_data=costNoAgent,
                        x_label='Numero iterazioni',
                        y_label='Costo magazzino', title='', label1='Costo con agente', label2='costo senza agente')
        plot_2d_graph(num_iteration_list, timeAgent, 'Numero iterazioni', 'Tempo esecuzione', '')
        plot_2d_graph(num_iteration_list, numTakenDecision, 'Numero iterazioni', 'Numero di decisioni prese', '')
        experiment_results['iterations'] = {'noAgent': costNoAgent, 'Agent': costAgent, 'timeAgent': timeAgent,
                                            'numDecision': numTakenDecision}

    if expLearningRateRL:
        n_rows = 3
        n_cols = 3
        time_limit = 100
        iterations = 2
        n_parcel_types = 5
        costNoAgent = []
        costAgent = []
        timeAgent = []
        numTakenDecision = []
        learningRateRL_list = [0.1, 0.3, 0.4, 0.5, 0.6, 0.9, 1]
        for alpha in learningRateRL_list:
            env = Warehouse(
                n_parcel_types=n_parcel_types,
                n_rows=n_rows,
                n_cols=n_cols
            )
            obs = env.reset()
            agent = Agent(warehouse=env, alpha=alpha, gamma=0.9, n_item=n_parcel_types, time_limit=time_limit, eps=0.3)
            costNoAgent.append(marshallingWithoutAgent(env, agent, time_limit))
            cost, time, no_empty_decision = marshallingWithAgentNN(env, agent, time_limit, iterations)
            costAgent.append(cost)
            timeAgent.append(time)
            numTakenDecision.append(no_empty_decision)

        plot_comparison(x_data=learningRateRL_list, y1_data=costAgent, y2_data=costNoAgent,
                        x_label='Learning Rate Agente',
                        y_label='Costo magazzino', title='', label1='Costo con agente', label2='costo senza agente')
        plot_2d_graph(learningRateRL_list, timeAgent, 'Learning Rate Agente', 'Tempo esecuzione', '')
        plot_2d_graph(learningRateRL_list, numTakenDecision, 'Learning Rate Agente', 'Numero di decisioni prese', '')
        experiment_results['LearningRate'] = {'noAgent': costNoAgent, 'Agent': costAgent, 'timeAgent': timeAgent,
                                              'numDecision': numTakenDecision}

    return experiment_results
