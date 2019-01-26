import itertools
import networkx as nx

nodesForPart2 = open('nodesForPart2.txt', 'r')
nodes = nodesForPart2.read().split('\n')
nodes = list(map(lambda x: int(x), nodes))
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(itertools.combinations(nodes, 2))
nx.write_edgelist(G, 'pairs.txt', delimiter=' ')