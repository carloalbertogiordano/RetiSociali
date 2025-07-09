from Graph.graph import Graph
import os
from cost_functions.factory import CostFunctionFactory as Cff
from cost_functions.factory import CostFuncType as Cft


def main():
    # Input parameters
    data_file = 'sourceData/facebook_data/facebook_combined.txt'
    max_budget = 10
    output_dir = 'results/'
    sub_graph_dim = 10

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # CSG: RANDOM, F1
    print("Running CSG with Cost: RANDOM, Goal: F1")
    test_name = "CSG_random_f1"
    print("#############################################################")
    fun =Cff.create_cost_function(Cft.RANDOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F1)
    print(f"Seed set (CSG, RANDOM, F1): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_RANDOM_F1.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: RANDOM, F2
    print("Running CSG with Cost: RANDOM, Goal: F2")
    test_name = "CSG_random_f2"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.RANDOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F2)
    print(f"Seed set (CSG, RANDOM, F2): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_RANDOM_F2.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: RANDOM, F3
    print("Running CSG with Cost: RANDOM, Goal: F3")
    test_name = "CSG_random_f3"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.RANDOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F3)
    print(f"Seed set (CSG, RANDOM, F3): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_RANDOM_F3.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: DEGREE, F1
    print("Running CSG with Cost: DEGREE, Goal: F1")
    test_name = "CSG_degree_f1"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.DEGREE)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F1)
    print(f"Seed set (CSG, DEGREE, F1): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_DEGREE_F1.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: DEGREE, F2
    print("Running CSG with Cost: DEGREE, Goal: F2")
    test_name = "CSG_degree_f2"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.DEGREE)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F2)
    print(f"Seed set (CSG, DEGREE, F2): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_DEGREE_F2.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: DEGREE, F3
    print("Running CSG with Cost: DEGREE, Goal: F3")
    test_name = "CSG_degree_f3"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.DEGREE)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F3)
    print(f"Seed set (CSG, DEGREE, F3): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_DEGREE_F3.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: CUSTOM, F1
    print("Running CSG with Cost: CUSTOM, Goal: F1")
    test_name = "CSG_custo_f1"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.CUSTOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F1)
    print(f"Seed set (CSG, CUSTOM, F1): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_CUSTOM_F1.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: CUSTOM, F2
    print("Running CSG with Cost: CUSTOM, Goal: F2")
    test_name = "CSG_custom_f2"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.CUSTOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F2)
    print(f"Seed set (CSG, CUSTOM, F2): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_CUSTOM_F2.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # CSG: CUSTOM, F3
    print("Running CSG with Cost: CUSTOM, Goal: F3")
    test_name = "CSG_custom_f3"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.CUSTOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('csg', select_goal_fun=Graph.GoalFuncType.F3)
    print(f"Seed set (CSG, CUSTOM, F3): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testCSG_CUSTOM_F3.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # WTSS: RANDOM
    print("Running WTSS with Cost: RANDOM")
    test_name = "WTSS_random"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.RANDOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('wtss')
    print(f"Seed set (WTSS, RANDOM): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testWTSS_RANDOM.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # WTSS: DEGREE
    print("Running WTSS with Cost: DEGREE")
    test_name = "WTSS_degree"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.DEGREE)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('wtss')
    print(f"Seed set (WTSS, DEGREE): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testWTSS_DEGREE.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")

    # WTSS: CUSTOM
    print("Running WTSS with Cost: CUSTOM")
    test_name = "WTSS_custom"
    print("#############################################################")
    fun = Cff.create_cost_function(Cft.CUSTOM)
    graph = Graph(data_file, max_budget, output_dir, fun, is_sub_graph=True, sub_graph_dim=sub_graph_dim)
    graph.calc_seed_set('wtss')
    print(f"Seed set (WTSS, CUSTOM): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")
    graph.calc_majority_cascade()
    graph.print_majority_cascade()
    graph.plot_majority_cascade()
    graph.save_plot('testWTSS_CUSTOM.png')
    graph.dyn_plot_cascade(test_name)
    print("#############################################################")


if __name__ == '__main__':
    main()