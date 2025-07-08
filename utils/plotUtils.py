import matplotlib.pyplot as plt

def plot_majority_cascade(cascade: list):
    x = list(range(len(cascade)))           # Indici: 0, 1, 2, ...
    y = [len(s) for s in cascade]           # Cardinalità di ogni set

    plt.plot(x, y, marker='o')
    plt.xlabel("Indice del passo nella cascata")
    plt.ylabel("Numero di nodi (len del set)")
    plt.title("Evoluzione della cascata")
    plt.grid(True)
    plt.show()