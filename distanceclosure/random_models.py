import networkx as nx
from distanceclosure.closure import distance_closure
import numpy as np


def degree_weighting_scale_free(N : int, m : int) -> nx.Graph:
    
    G = nx.barabasi_albert_graph(N, m)
    E2 = 4*G.number_of_edges()*G.number_of_edges()
    
    for u, v in G.edges():
        G[u][v]['proximity'] = G.degree(u)*G.degree(v)/E2
        G[u][v]['distance'] = 1./G[u][v]['proximity'] - 1
    
    return G

def degree_weighting_random_graph(N, p):
    G = nx.fast_gnp_random_graph(N, p)
    E2 = 4*G.number_of_edges()*G.number_of_edges()
    
    for u, v in G.edges():
        G[u][v]['proximity'] = G.degree(u)*G.degree(v)/E2
        G[u][v]['distance'] = 1./G[u][v]['proximity'] - 1
    
    return G

def lognormal_distortion_weighting(N, m, tau=0.5, mu=2.75, sigma=2.8, structure='scale_free'):
    '''
    N : Number of nodes
    m : Number of connections from new nodes
    tau : relative backbone size
    mu : average log-distortion
    sigma : st. deviation log-distortion
    '''

    if structure == 'scale_free':
        G = nx.barabasi_albert_graph(N, m)
    elif structure == 'random_graph':
        G = nx.fast_gnp_random_graph(N, 2*m/N) # Make sure one component only
    else: # Implement configuration model
        raise NotImplementedError

    nx.set_edge_attributes(G, 1, name='distance')
    nx.set_edge_attributes(G, 0.5, name='weight')
    nx.set_edge_attributes(G, 1, name='s_value')

    E = int((1./tau - 1.)*G.number_of_edges())
    s_values = np.random.lognormal(mean=mu, sigma=sigma, size=E)

    newG = _from_distortion_distribution(G, s_values+1)

    return newG, s_values

def _from_distortion_distribution(D, s_values, weight='distance', kind='metric'):
    
    G = D.copy()

    G_closure = distance_closure(G, kind=kind, weight=weight, only_backbone=False)
    G_closure.remove_edges_from(G.edges())
    semi_edges = list(G_closure.edges())
    np.random.shuffle(semi_edges)

    kind_distance = '{kind:s}_distance'.format(kind=kind)
    for idx, (u, v) in enumerate(semi_edges[:len(s_values)]):
        d = s_values[idx]*G_closure[u][v][kind_distance]
        p = 1./(d+1.)
        G.add_edge(u, v, **{weight: d, 'weight': p, 's_value':s_values[idx]})

    return G


def random_symmetry_breaking(G, alpha=0.5):
    GD = G.to_directed()
    nx.set_edge_attributes(GD, values=0, name='asymmetry')
    
    for u, v, d in GD.edges(data=True):
        if d['asymmetry'] == 0:
            GD[u][v]['asymmetry'] = alpha if np.random.rand() < 0.5 else -alpha
            GD[v][u]['asymmetry'] = -GD[u][v]['asymmetry']
            
            GD[v][u]['proximity'] = ((1-GD[u][v]['asymmetry'])/(1+GD[u][v]['asymmetry']))*GD[u][v]['proximity']
            GD[v][u]['distance'] = 1./GD[v][u]['proximity'] - 1
    
    return GD