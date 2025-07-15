import os
import yaml
from multiprocessing import Process, set_start_method

from Graph.graph import Graph
from cost_functions.factory import CostFunctionFactory as Cff
from cost_functions.factory import CostFuncType as Cft
from Graph.graph import GoalFuncType as Gft  # Supponendo che esista così


# ----------------- Config Loader -----------------
def load_config(path='config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


# ----------------- Enum String Mapping -----------------
def str_to_cost_func(s):
    return getattr(Cft, s.upper())


def str_to_goal_func(s):
    return getattr(Gft, s.upper()) if s else None


# ---------- Algorithm Runner ----------
def run_algorithm(
    algorithm_type, cost_type, goal_type, info_name, data_file, output_dir,
    sub_graph_dim, saved_graph, enable_vis, genetic_params=None
):
    print(f"run_algo called with params: {algorithm_type} {cost_type}, {goal_type}, {info_name}, {data_file}, {output_dir}, {sub_graph_dim}, {saved_graph}, {enable_vis}, {genetic_params}")

    cost_func = Cff.create_cost_function(cost_type) if cost_type else None
    graph = Graph(data_file, output_dir, cost_func,
                  is_sub_graph=True, sub_graph_dim=sub_graph_dim, info_name=info_name)
    graph.set_graph(saved_graph)

    print(f"[{algorithm_type.upper()}] Cost: {cost_type.name if cost_type else 'None'}, "
          f"Goal: {goal_type.name if goal_type else 'None'}")

    if algorithm_type == 'genetic':
        # Passiamo i parametri al genetic_search
        graph.genetic_search(select_goal_fun=goal_type, **(genetic_params or {}))
    else:
        graph.calc_seed_set(algorithm_type, select_goal_fun=goal_type)

    print(f"Seed set ({algorithm_type.upper()}): {graph.get_seed_set()} "
          f"(size={len(graph.get_seed_set())})")

    graph.calc_majority_cascade()
    #graph.print_majority_cascade()
    graph.save_cascade_as_json()
    graph.plot_majority_cascade()

    if enable_vis:
        graph.plot_majority_cascade()
        graph.save_plot(f'{info_name}_plot.png')
        graph.dyn_plot_cascade()


# ----------------- Main Function -----------------
def main():
    config = load_config()

    # Graph global settings
    graph_config = config['graph']
    data_file = graph_config['file_path']
    output_dir = graph_config['save_path']
    sub_graph_dim = graph_config['sub_graph_dim']
    base_info_name = graph_config.get('info_name', 'experiment')
    enable_vis = graph_config.get('cascade_visualization', True)
    is_sub_graph = graph_config.get('is_sub_graph', True)

    # Multiprocessing config
    use_multiprocessing = config.get('use_multiprocessing', True)

    # Load genetic params
    genetic_params = config.get('genetic_parameters', {})

    if use_multiprocessing:
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass

    os.makedirs(output_dir, exist_ok=True)

    # Create reusable base graph
    base_cost = Cff.create_cost_function(Cft.RANDOM)
    base_graph = Graph(data_file, output_dir, base_cost, is_sub_graph=is_sub_graph, sub_graph_dim=sub_graph_dim)
    saved_graph = base_graph.get_graf()

    processes = []

    for algo_type, task_list in config['algorithms'].items():
        print(f"Ruinning {algo_type}")
        for idx, task in enumerate(task_list):
            print(f"with task {idx}:{task}")
            cost_type = str_to_cost_func(task['cost_type']) if task.get('cost_type') else None
            goal_type = str_to_goal_func(task.get('goal_type'))

            goal_name = goal_type.name.lower() if goal_type else "none"
            cost_name = cost_type.name.lower() if cost_type else "none"
            # Include base_info_name in the task name
            info_name = f"{base_info_name}_{algo_type.upper()}_{cost_name}_{goal_name}"

            args = (
                algo_type, cost_type, goal_type,
                info_name, data_file, output_dir, sub_graph_dim,
                saved_graph, enable_vis, genetic_params
            )

            print(f"Calling with args: {args}")

            if use_multiprocessing:
                p = Process(target=run_algorithm, args=args)
                p.start()
                processes.append(p)
            else:
                run_algorithm(*args)

    if use_multiprocessing:
        for p in processes:
            p.join()
        print("\n ---> All processes completed.")


if __name__ == '__main__':
    main()
