import json
import matplotlib.pyplot as plt
from pathlib import Path
from cost_functions.factory import CostFunctionFactory as CFF
from cost_functions.factory import CostFuncType as CFT
from Graph.graph import Graph



def get_file_list(directory: str, extension: str = ".json"):
    """
    Return a list of all files in `directory` with the given `extension`.
    """
    d = Path(directory)
    return [f for f in d.iterdir() if f.is_file() and f.suffix == extension]


def plot_majority_cascade(cascade_list: list, metadata_list: list, fig_title: str) -> None:
    """
    Plot multiple majority cascades, each labeled by title in the legend,
    and draw a separate info‑box showing Spread/Cost/Ratio for each algorithm,
    anchored to the bottom‑right margin, with automatic height estimation
    and enough space for the axes so the plot isn't squeezed left.
    """
    # 1) Create a wider figure so there's ample room for the plot and the margin
    fig, ax = plt.subplots(figsize=(14, 6))  # increase width to 14"

    # 2) Plot each cascade line
    for cascade, meta in zip(cascade_list, metadata_list):
        title = meta["title"]
        steps = cascade[0] if isinstance(cascade, list) and cascade else []
        if not all(isinstance(step, list) for step in steps):
            print(f"Invalid format for '{title}', skipping.")
            continue
        sizes = [len(step) for step in steps]
        x = list(range(len(sizes)))
        ax.plot(x, sizes, marker='o', label=title)

    # 3) Standard axis labels, title, and grid
    ax.set_xlabel("Cascade Step Index")
    ax.set_ylabel("Number of Nodes")
    ax.set_title(f"Majority Cascade on: {fig_title}")
    ax.grid(True)

    # 4) Place the legend outside on the upper‑right corner
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))

    # 5) Build the multiline info text for Spread/Cost/Ratio
    info_lines = [
        f"{meta['title']}:\n- Spread={meta['spread']}\n- Cost={meta['cost']}\n- Ratio={meta['ratio']}"
        for meta in metadata_list
    ]
    info_text = "\n".join(info_lines)

    # 6) Estimate the height of the info‑box
    tmp = ax.text(
        1.02, 0.0, info_text,
        transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.5')
    )
    fig.canvas.draw()
    renderer = fig.canvas.renderer
    bbox = tmp.get_window_extent(renderer)
    inv = ax.transAxes.inverted()
    bbox_axes = bbox.transformed(inv)
    text_height = bbox_axes.height
    tmp.remove()

    # 7) Draw the real info‑box at the bottom‑right
    ax.text(
        1.02, 0.0, info_text,
        transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray')
    )

    # 8) Adjust subplot so axes take full available space and leave room on right
    #    for legend and info‑box. Right=0.75 leaves 25% margin for them.
    fig.subplots_adjust(right=0.75)

    plt.show()



def extract_parameters(json_data: dict, reference_file: str) -> dict:
    """
    Given the loaded JSON data, compute and return:
      - spread:   |inf, t| - |inf, 0|
      - cost:     cost_fun([inf, 0])
      - ratio:    spread / cost
    Adjust this logic to your real definitions of Spread, Cost, Ratio.
    """

    cost_fun_val = -1
    cost_fun_names = {"RANDOM":CFT.RANDOM, "CUSTOM":CFT.CUSTOM, "DEGREE":CFT.DEGREE}
    upper_filename = reference_file.upper()
    graph = None
    inf_0 = json_data[0]
    inf_t = json_data[-1]


    # ---- CALCULATE COST VALUE -----
    # 1) Find which cost function name appears in the filename
    for cost in cost_fun_names.keys():
        if cost in upper_filename:
            cost_fun_val = cost_fun_names[cost]
            break

    cost_fun = CFF.create_cost_function(cost_fun_val)

    graph = Graph("../sourceData/facebook_data/facebook_combined.txt", "None", cost_fun,
                  is_sub_graph=False, sub_graph_dim=0, info_name="None")


    cost = 0
    for node in inf_0:
        cost += cost_fun.calculate_cost(graph=graph, node_label=node)
    # ---------------------------------


    print(f"|Inf, t| = {len(inf_t)}")
    print(f"|Inf, 0| = {len(inf_0)}")
    print(f"|V| = {len(graph.get_nodes_list())}")
    print(f"budget = {graph.get_budget()}")
    print(f"cost = {cost}")

    return {
        "spread": len(inf_t)-len(inf_0),
        "cost":   cost,
        "ratio":  (len(inf_t)-len(inf_0))/len(graph.get_nodes_list()) * (graph.budget / cost)
    }


def main():
    json_dir = "plot_jsons/"
    files = get_file_list(json_dir)

    all_cascades = []
    metadata = []

    for f in files:
        name = f.stem
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Error loading '{name}': {e}")
            continue

        # store the cascade data
        all_cascades.append(data)

        # compute and attach parameters
        params = extract_parameters(data[0], name)
        params["title"] = name
        metadata.append(params)

    plot_majority_cascade(all_cascades, metadata, "CSG Custom")


if __name__ == "__main__":
    main()
