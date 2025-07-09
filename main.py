from Graph.graph import Graph
import os


def main():
    # Input parameters
    data_file = 'sourceData/facebook_data/facebook_combined.txt'
    max_nodes = 80
    output_dir = 'results/'
    sub_graph_dim = 100

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # CSG: RANDOM, F1
    print("Running CSG with Cost: RANDOM, Goal: F1")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.RANDOM, select_goal_fun=Graph.GoalFuncType.F1)
    print(f"Seed set (CSG, RANDOM, F1): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_RANDOM_F1.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: RANDOM, F2
    print("Running CSG with Cost: RANDOM, Goal: F2")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.RANDOM, select_goal_fun=Graph.GoalFuncType.F2)
    print(f"Seed set (CSG, RANDOM, F2): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_RANDOM_F2.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: RANDOM, F3
    print("Running CSG with Cost: RANDOM, Goal: F3")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.RANDOM, select_goal_fun=Graph.GoalFuncType.F3)
    print(f"Seed set (CSG, RANDOM, F3): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_RANDOM_F3.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: DEGREE, F1
    print("Running CSG with Cost: DEGREE, Goal: F1")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.DEGREE, select_goal_fun=Graph.GoalFuncType.F1)
    print(f"Seed set (CSG, DEGREE, F1): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_DEGREE_F1.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: DEGREE, F2
    print("Running CSG with Cost: DEGREE, Goal: F2")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.DEGREE, select_goal_fun=Graph.GoalFuncType.F2)
    print(f"Seed set (CSG, DEGREE, F2): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_DEGREE_F2.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: DEGREE, F3
    print("Running CSG with Cost: DEGREE, Goal: F3")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.DEGREE, select_goal_fun=Graph.GoalFuncType.F3)
    print(f"Seed set (CSG, DEGREE, F3): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_DEGREE_F3.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: CUSTOM, F1
    print("Running CSG with Cost: CUSTOM, Goal: F1")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.CUSTOM, select_goal_fun=Graph.GoalFuncType.F1)
    print(f"Seed set (CSG, CUSTOM, F1): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_CUSTOM_F1.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: CUSTOM, F2
    print("Running CSG with Cost: CUSTOM, Goal: F2")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.CUSTOM, select_goal_fun=Graph.GoalFuncType.F2)
    print(f"Seed set (CSG, CUSTOM, F2): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_CUSTOM_F2.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # CSG: CUSTOM, F3
    print("Running CSG with Cost: CUSTOM, Goal: F3")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_cost=Graph.CostFuncType.CUSTOM, select_goal_fun=Graph.GoalFuncType.F3)
    print(f"Seed set (CSG, CUSTOM, F3): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_CUSTOM_F3.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # WTSS: RANDOM
    print("Running WTSS with Cost: RANDOM")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('wtss', select_cost=Graph.CostFuncType.RANDOM)
    print(f"Seed set (WTSS, RANDOM): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testWTSS_RANDOM.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # WTSS: DEGREE
    print("Running WTSS with Cost: DEGREE")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('wtss', select_cost=Graph.CostFuncType.DEGREE)
    print(f"Seed set (WTSS, DEGREE): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testWTSS_DEGREE.png')
    graph.dyn_plot_cascade()
    print("#############################################################")

    # WTSS: CUSTOM
    print("Running WTSS with Cost: CUSTOM")
    print("#############################################################")
    graph = Graph(data_file, max_nodes, output_dir, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('wtss', select_cost=Graph.CostFuncType.CUSTOM)
    print(f"Seed set (WTSS, CUSTOM): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testWTSS_CUSTOM.png')
    graph.dyn_plot_cascade()
    print("#############################################################")


if __name__ == '__main__':
    main()