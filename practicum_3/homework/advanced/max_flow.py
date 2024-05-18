from typing import Union
import matplotlib.pyplot as plt
import networkx as nx
import time

def plot_graph(
        G: Union[nx.Graph, nx.DiGraph], highlighted_edges: list[tuple[any, any]] = None
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    pos = nx.spring_layout(G)
    edge_color_list = ["black"] * len(G.edges)
    if highlighted_edges:
        for i, edge in enumerate(G.edges()):
            if edge in highlighted_edges or (edge[1], edge[0]) in highlighted_edges:
                edge_color_list[i] = "red"
    options = dict(
        font_size=12,
        node_size=500,
        node_color="white",
        edgecolors="black",
        edge_color=edge_color_list,
    )
    nx.draw_networkx(G, pos, ax=ax, **options)
    if nx.is_weighted(G):
        labels = {e: G.edges[e]["weight"] for e in G.edges}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels)
    plt.show()


def dfs(C, F, s, t, Max_flow, Now_path):
    Now_path.append(s)
    if s == t:
        return Max_flow, Now_path
    for i in range(len(C)):
        propusk_sposop = C[s][i] - F[s][i]
        if propusk_sposop > 0 and i not in Now_path:
            min_flow, result_path = dfs(C, F, i, t, min(Max_flow, propusk_sposop), Now_path)
            if min_flow > 0:
                return min_flow, result_path
    Now_path.pop()
    return 0, []


def max_flow(G: nx.Graph, s: any, t: any) -> int:
    size = max([int(i) for i in G.nodes()]) + 1
    C = [[0] * size for i in range(size)]
    F = [[0] * size for i in range(size)]
    for u, v, data in G.edges(data=True):
        C[int(u)][int(v)] = data['weight']
    value: int = 0
    while True:
        flow, path = dfs(C, F, s, t, float('Inf'), [])
        if flow == 0:
            break
        value += flow
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            F[u][v] += flow
            F[v][u] -= flow
    return value


if __name__ == "__main__":
    path = "/Users/ignat/Desktop/pershin_homework/spbu-fundamentals-of-algorithms/practicum_3/homework/advanced/graph_1.edgelist"
    G = nx.read_edgelist(path, create_using=nx.DiGraph)
    # plot_graph(G)
    start = time.time()
    val = max_flow(G, s=0, t=5)
    print(time.time() - start)
    print(f"Maximum flow is {val}. Should be 23")
