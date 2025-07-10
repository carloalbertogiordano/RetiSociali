import os
from multiprocessing import Process, set_start_method
from Graph.graph import Graph
from cost_functions.factory import CostFunctionFactory as Cff
from cost_functions.factory import CostFuncType as Cft

# Switch between sequential and parallel execution
USE_MULTIPROCESSING = True

def run_csg(cost_type, goal_type, test_name, data_file, output_dir, sub_graph_dim, saved_graph):
    print(f"[CSG] Cost: {cost_type.name}, Goal: {goal_type.name}")
    cost_func = Cff.create_cost_function(cost_type)
    graph = Graph(data_file, output_dir, cost_func, is_sub_graph=True, sub_graph_dim=sub_graph_dim, info_name=test_name)
    graph.set_graph(saved_graph)
    graph.calc_seed_set('csg', select_goal_fun=goal_type)
    print(f"Seed set (CSG, {cost_type.name}, {goal_type.name}): {graph.get_seed_set()} (size={len(graph.get_seed_set())})")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot(f'testCSG_{cost_type.name}_{goal_type.name}.png')
    graph.dyn_plot_cascade()

def run_wtss(cost_type, test_name, data_file, output_dir, sub_graph_dim, saved_graph):
    print(f"[WTSS] Cost: {cost_type.name}")
    cost_func = Cff.create_cost_function(cost_type)
    graph = Graph(data_file, output_dir, cost_func, is_sub_graph=True, sub_graph_dim=sub_graph_dim, info_name=test_name)
    graph.set_graph(saved_graph)
    graph.calc_seed_set('wtss')
    print(f"Seed set (WTSS, {cost_type.name}): {graph.get_seed_set()} (size={len(graph.get_seed_set())})")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot(f'testWTSS_{cost_type.name}.png')
    graph.dyn_plot_cascade()

def run_genetic(cost_type, goal_type, test_name, data_file, output_dir, sub_graph_dim, saved_graph):
    cost_name = cost_type.name if cost_type else "CASCADE"
    goal_name = goal_type.name if goal_type else "CASCADE"
    print(f"[GENETIC] Cost: {cost_name}, Goal: {goal_name}")
    cost_func = Cff.create_cost_function(cost_type) if cost_type else None
    graph = Graph(data_file, output_dir, cost_func, is_sub_graph=True, sub_graph_dim=sub_graph_dim, info_name=test_name)
    graph.set_graph(saved_graph)
    graph.calc_seed_set('genetic', select_goal_fun=goal_type)  # goal_type può essere None
    print(f"Seed set (GENETIC, {cost_name}, {goal_name}): {graph.get_seed_set()} (size={len(graph.get_seed_set())})")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot(f'testGENETIC_{cost_name}_{goal_name}.png')
    graph.dyn_plot_cascade()


def main():
    if USE_MULTIPROCESSING:
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

    data_file = 'sourceData/facebook_data/facebook_combined.txt'
    output_dir = 'results/'
    sub_graph_dim = 200
    os.makedirs(output_dir, exist_ok=True)

    # Build the base graph only once and share its structure
    base_cost = Cff.create_cost_function(Cft.RANDOM)
    base_graph = Graph(data_file, output_dir, base_cost, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    saved_graph = base_graph.get_graf()

    csg_tasks = [
        #(Cft.RANDOM, Graph.GoalFuncType.F1),
        #(Cft.RANDOM, Graph.GoalFuncType.F2),
        #(Cft.RANDOM, Graph.GoalFuncType.F3),
        #(Cft.DEGREE, Graph.GoalFuncType.F1),
        #(Cft.DEGREE, Graph.GoalFuncType.F2),
        #(Cft.DEGREE, Graph.GoalFuncType.F3),
        #(Cft.CUSTOM, Graph.GoalFuncType.F1),
        #(Cft.CUSTOM, Graph.GoalFuncType.F2),
        #(Cft.CUSTOM, Graph.GoalFuncType.F3),
    ]

    wtss_tasks = [] #[(Cft.RANDOM,), (Cft.DEGREE,), (Cft.CUSTOM,)]

    genetic_tasks = [
        (Cft.RANDOM, Graph.GoalFuncType.F1),
        (Cft.RANDOM, Graph.GoalFuncType.F2),
        (Cft.RANDOM, Graph.GoalFuncType.F3),
        (Cft.RANDOM, None),
        (Cft.DEGREE, Graph.GoalFuncType.F1),
        (Cft.DEGREE, Graph.GoalFuncType.F2),
        (Cft.DEGREE, Graph.GoalFuncType.F3),
        (Cft.DEGREE, None),
        (Cft.CUSTOM, Graph.GoalFuncType.F1),
        (Cft.CUSTOM, Graph.GoalFuncType.F2),
        (Cft.CUSTOM, Graph.GoalFuncType.F3),
        (Cft.CUSTOM, None),
    ]

    processes = []

    # Launch CSG tasks
    for cost_type, goal_type in csg_tasks:
        name = f"CSG_{cost_type.name.lower()}_{goal_type.name.lower()}"
        args = (cost_type, goal_type, name, data_file, output_dir, sub_graph_dim, saved_graph)
        if USE_MULTIPROCESSING:
            p = Process(target=run_csg, args=args)
            p.start()
            processes.append(p)
        else:
            run_csg(*args)

    # Launch WTSS tasks
    for (cost_type,) in wtss_tasks:
        name = f"WTSS_{cost_type.name.lower()}"
        args = (cost_type, name, data_file, output_dir, sub_graph_dim, saved_graph)
        if USE_MULTIPROCESSING:
            p = Process(target=run_wtss, args=args)
            p.start()
            processes.append(p)
        else:
            run_wtss(*args)

    # Launch GENETIC tasks
    for cost_type, goal_type in genetic_tasks:
        goal_name = goal_type.name.lower() if goal_type else 'none'
        name = f"GENETIC_{cost_type.name.lower()}_{goal_name}"
        args = (cost_type, goal_type, name, data_file, output_dir, sub_graph_dim, saved_graph)
        if USE_MULTIPROCESSING:
            p = Process(target=run_genetic, args=args)
            p.start()
            processes.append(p)
        else:
            run_genetic(*args)

    if USE_MULTIPROCESSING:
        for p in processes:
            p.join()
        print("\n ---> All processes completed.")

if __name__ == '__main__':
    main()
