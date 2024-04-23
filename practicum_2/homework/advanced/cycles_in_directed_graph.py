import os

import networkx as nx

TEST_GRAPH_FILES = ["graph_1_wo_cycles.edgelist", "graph_2_wo_cycles.edgelist", "graph_3_w_cycles.edgelist"]

def has_cycles(g: nx.DiGraph):
    rec_stack = set()
    vis = set()

    def dfs(node):
        if node in rec_stack:
            return True

        if node in vis:
            return False

        vis.add(node)
        rec_stack.add(node)
        for neighbor in g.neighbors(node):
            if dfs(neighbor):
                return True

        rec_stack.remove(node)
        return False

    for node in g.nodes():
        if dfs(node):
            return True
    return False


if __name__ == "__main__":
    for filename in TEST_GRAPH_FILES:
        # Load the graph
        G = nx.read_edgelist(f"practicum_2/homework/advanced/{filename}", create_using=nx.DiGraph)
        G = nx.read_edgelist(
            os.path.join("practicum_2", "homework", filename), create_using=nx.DiGraph
        )
        # Output whether it has cycles
        print(f"Graph {filename} has cycles: {has_cycles(G)}")
