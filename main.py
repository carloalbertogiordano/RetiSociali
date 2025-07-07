import igraph as ig
import os

def main():
    filepath = os.path.join(os.path.dirname(__file__), 'graphs/facebook_data/facebook_combined.txt')
    g = ig.Graph.Read_Edgelist(filepath, directed=False)

    print(f"Grafo caricato: {g.vcount()} nodi, {g.ecount()} archi.")

    # Estrai la componente più grande
    largest_cc = ig.Graph.connected_components(g)

    # Per motivi di visualizzazione, prendiamo un campione di 300 nodi
    subgraph = largest_cc.induced_subgraph(list(range(300)))

    print(f"Sottografo per la visualizzazione: {subgraph.vcount()} nodi, {subgraph.ecount()} archi.")
    # Plot con layout automatico
    layout = subgraph.layout("fr")  # Fruchterman-Reingold
    ig.plot(subgraph, layout=layout, vertex_size=10, vertex_label=None, bbox=(800, 800))

if __name__ == '__main__':
    main()
