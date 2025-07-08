import igraph as ig
import os
import random


def get_subgraph(graph: ig.Graph, number: int):
    return graph.induced_subgraph(list(range(number)))


def save_plot(graph: ig.Graph, file_path: str, vertex_number=0):
    if vertex_number == 0:
        vertex_number = graph.vcount()
        subgraph = graph
    else:
        subgraph = get_subgraph(graph, vertex_number)

    layout = graph.layout("fr")  # Fruchterman-Reingold

    ig.plot(
        subgraph,
        target=file_path,
        layout=layout,
        vertex_size=10,
        vertex_label=None,
        bbox=(1000, 1000),
    )
