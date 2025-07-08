from Graph.graph import Graph
import os

def main():
    graph = Graph('graphs/facebook_data/facebook_combined.txt',
                  100,
                  'graphs/',
                  is_sub_graph=True,
                  sub_graph_dim=100)
    graph.calc_seed_set('csg',
                        select_cost=Graph.CostFuncType.RANDOM,
                        select_goal_fun=Graph.GoalFuncType.F1)

    print(f"Seed set (CSG): {graph.get_seed_set()}")

    graph.calc_majority_cascade()
    graph.print_majority_cascade()

    graph.plot_majority_cascade()

    graph.save_plot('testCSG.png')

    print("#############################################################")

    '''graph.calc_seed_set('wtss', select_cost=2)
    print(graph.get_seed_set())

    graph.calc_majority_cascade()
    graph.print_majority_cascade()

    graph.plot_majority_cascade()

    graph.save_plot('testWTSS.png')

    print("#############################################################")'''

if __name__ == '__main__':
    main()
