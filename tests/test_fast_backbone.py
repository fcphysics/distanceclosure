import networkx as nx
import distanceclosure as dc

# Instanciate a (weighted) graph
edgelist = {
    ('s', 'a'): 8,
    ('s', 'c'): 6,
    ('s', 'd'): 5,
    ('a', 'd'): 2,
    ('a', 'e'): 1,
    ('b', 'e'): 6,
    ('c', 'd'): 3,
    ('c', 'f'): 9,
    ('d', 'f'): 4,
    ('e', 'g'): 4,
    ('f', 'g'): 0,
}
G = nx.from_edgelist(edgelist)
# Make sure every edge has an attribute with the distance value
nx.set_edge_attributes(G, name='distance', values=edgelist)

# Compute backbone and semi-metric distortions
#B, s_vals = dc.metric_backbone_distortion(G, weight='distance')
B = dc.metric_backbone(G, weight='distance')

print(G.number_of_edges())
print(B.number_of_edges())
#print(s_vals)