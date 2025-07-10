import os
import matplotlib.pyplot as plt
import igraph as ig
from matplotlib import cm
from PIL import Image
import glob


def save_plot(graph, filename: str):
    """
    Save a visual representation of the graph to a file.

    :param graph:
    :param filename: The name of the file to save the plot (without path).
    """
    layout = graph.graph.layout("fr")  # Fruchterman-Reingold

    ig.plot(
        graph.graph,
        target=graph.save_path + 'images/' + filename,
        layout=layout,
        vertex_size=10,
        vertex_label=None,
        bbox=(2000, 2000),
    )


def plot_majority_cascade(graph):
    """
    Plot the evolution of the majority cascade over time.
    """
    x = list(range(len(graph.cascade)))  # Indici: 0, 1, 2, ...
    y = [len(s) for s in graph.cascade]  # Cardinalità di ogni set

    plt.plot(x, y, marker='o')
    plt.xlabel("Indice del passo nella cascata")
    plt.ylabel("Numero di nodi (len del set)")
    plt.title(f"Evoluzione della cascata per {graph.info_name}")
    plt.grid(True)
    plt.show()


def dyn_plot_cascade(graph):
    """
    Generates a sequence of network plots representing the cumulative activation of nodes
    over the steps of a cascade process, and compiles them into an animated GIF.

    The cascade is visualized using color intensity to indicate activation time, and
    non-activated nodes are shown in gray. All output images and the final GIF are saved
    under a structured directory inside `graph.save_path`.

    Output structure:
    └── graph.save_path/
        └── plots/
            └── plot_cascade/
                ├── images/
                │   ├── step_00.png
                │   ├── step_01.png
                │   └── ...
                └── diffusione.gif
    """
    return
    layout = graph.graph.layout("fr")  # Use force-directed layout for graph positioning
    max_step = len(graph.get_majority_cascade())
    colormap = cm.get_cmap("plasma", max_step + 1)

    def rgba_to_hex(rgba):
        """Convert RGBA color to HEX string (ignore alpha)."""
        r, g, b, _ = rgba
        return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))

    # === Centralized directory structure ===
    base_output_dir = os.path.join(graph.save_path, "plots")
    cascade_dir = os.path.join(base_output_dir, "plot_cascade")
    images_dir = os.path.join(cascade_dir, "images")
    gif_path = os.path.join(cascade_dir, f"diffusione_{graph.info_name}.gif")

    # Create output directories if they do not exist
    os.makedirs(images_dir, exist_ok=True)

    # Compute cumulative activated nodes at each cascade step
    cumulative_cascade = []
    active_nodes = set()
    for step in graph.get_majority_cascade():
        active_nodes |= step
        cumulative_cascade.append(active_nodes.copy())

    # Generate one image per step
    for t, active in enumerate(cumulative_cascade):
        colors = []
        for v in range(graph.graph.vcount()):
            if v in active:
                c = colormap(t)  # Color based on activation time
                c_hex = rgba_to_hex(c)
            else:
                c_hex = "#dddddd"  # Gray for inactive nodes
            colors.append(c_hex)

        ig.plot(
            graph.graph,
            target=os.path.join(images_dir, f"step_{t:02d}_{graph.info_name}.png"),
            layout=layout,
            vertex_color=colors,
            vertex_size=8,
            bbox=(3000, 3000),
            margin=40,
        )

    # Load all generated images and compile into an animated GIF
    images = [Image.open(f) for f in sorted(glob.glob(os.path.join(images_dir, f"step_*{graph.info_name}.png")))]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1000,  # Frame duration in milliseconds
        loop=0
    )
