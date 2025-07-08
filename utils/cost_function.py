import igraph as ig
import random


def cost1(node: ig.Vertex, rangeLow=1, rangeMax=10):
    return random.randint(rangeLow, rangeMax)


def cost2(node: ig.Vertex):
    return node.degree() / 2


def cost3(node: ig.Vertex):
    return cost1(node, 10, 1000)  # TODO: Creare una funzione ad hoc
