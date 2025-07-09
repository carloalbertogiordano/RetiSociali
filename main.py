from Graph.graph import Graph
import os


def main():
    """graph = Graph('graphs/facebook_data/facebook_combined.txt',
                  350,
                  'graphs/',
                  is_sub_graph=True,
                  sub_graph_dim=700)
    graph.calc_seed_set('csg',
                        select_cost=Graph.CostFuncType.RANDOM,
                        select_goal_fun=Graph.GoalFuncType.F1)

    print(f"Seed set (CSG): {graph.get_seed_set()}")

    graph.calc_majority_cascade()

    print("#############################################################")
    graph.print_majority_cascade()

    graph.plot_majority_cascade()

    graph.save_plot('testCSG.png')"""

    print("#############################################################")

    graph = Graph('sourceData/facebook_data/facebook_combined.txt',
                  800,
                  'results/',
                  is_sub_graph=True,
                  sub_graph_dim=1000)
    graph.calc_seed_set('wtss', select_cost=Graph.CostFuncType.RANDOM)
    print(
        f"Seed set (CSG): {graph.get_seed_set()}; |Seed set|: {len(graph.get_seed_set())}\n")

    graph.calc_majority_cascade()
    graph.print_majority_cascade()

    graph.plot_majority_cascade()

    graph.save_plot('testWTSS.png')
    
    graph.dyn_plot_cascade()

    print("#############################################################")


if __name__ == '__main__':
    main()
