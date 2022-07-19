# -*- coding: utf-8 -*-
"""
Transitive Closure
==================

Computes transitive closure on a weighted graph.
These algorithms work with undirected weighted (distance) graphs.
"""

import numpy as np
import networkx as nx
from distanceclosure.dijkstra import all_pairs_dijkstra_path_length
__name__ = 'distanceclosure'
__author__ = """\n""".join(['Rion Brattig Correia <rionbr@gmail.com>', 'Felipe Xavier Costa <fcosta@binghamton.com>'])

__all__ = [
    "distance_closure",
    "s_values",
    "b_values"
]


__kinds__ = ['metric', 'ultrametric', 'product', 'minkowski']
__algorithms__ = ['dense', 'dijkstra']


#### FROM - closure.py ####

def distance_closure(D, kind='metric', minkowski_par=1.0, algorithm='dijkstra', weight='weight', only_backbone=False, verbose=False, *args, **kwargs):
    """Computes the transitive closure (All-Pairs-Shortest-Paths; APSP)
    using different shortest path measures on the distance graph
    (adjacency matrix) with values in the ``[0,inf]`` interval.

    .. math::

        c_{ij} = min_{k}( metric ( a_{ik} , b_{kj} ) )

    Parameters
    ----------
    D : NetworkX.Graph
        The Distance graph.

    kind : string
        Type of closure to compute: ``metric`` or ``ultrametric``.

    algorithm : string
        Type of algorithm to use: ``dense`` or ``dijkstra``.

    weight : string
        Edge property containing distance values. Defaults to `weight`.
    
    only_backbone : bool
        Only include new distance closure values for edges in the original graph.
    
    Verbose :bool
        Prints statements as it computes.

    Returns
    --------
    C : NetworkX.Graph
        The distance closure graph. Note this may be a fully connected graph.

    Examples
    --------
    >>> distance_closure(D, kind='metric', algorithm='dijkstra', weight='weight', only_backbone=True)

    Note
    ----
    Dense matrix is slow for large graphs.
    We are currently working on optimizing it.
    If your network is large and/or sparse, use the Dijkstra method.

    - Metric: :math:`(min,+)`
    - Ultrametric: :math:`(min,max)` -- also known as maximum flow.
    - Semantic proximity: (to be implemented)

    .. math::

            [ 1 + \\sum_{i=2}^{n-1} log k(v_i) ]^{-1}
    """
    _check_for_kind(kind)
    _check_for_algorithm(algorithm)

    G = D.copy()

    # Dense
    if algorithm == 'dense':

        raise NotImplementedError('Needs some fine tunning.')
        #M = nx.to_numpy_matrix(D, *args, **kwargs)
        #return _transitive_closure_dense_numpy(M, kind, *args, **kwargs)

    # Dijkstra
    elif algorithm == 'dijkstra':

        if kind == 'metric':
            disjunction = sum
        elif kind == 'ultrametric':
            disjunction = max
        elif kind == 'product':
            disjunction = prod
        elif kind == 'minkowski':
            mink.__defaults__ = ([0, 1], minkowski_par)
            disjunction = mink

        edges_seen = set()
        i = 1
        total = G.number_of_nodes()
        # APSP
        for u, lengths in all_pairs_dijkstra_path_length(G, weight=weight, disjunction=disjunction):
            if verbose:
                per = i / total
                print("Closure: Dijkstra : {kind:s} : source node {u:s} : {i:d} of {total:d} ({per:.2%})".format(kind=kind, u=u, i=i, total=total, per=per))
            for v, length in lengths.items():

                if (u, v) in edges_seen or u == v:
                    continue
                else:
                    edges_seen.add((u, v))
                    kind_distance = '{kind:s}_distance'.format(kind=kind)
                    is_kind = 'is_{kind:s}'.format(kind=kind)
                    if not G.has_edge(u, v):
                        if not only_backbone:
                            G.add_edge(u, v, **{weight: np.inf, kind_distance: length})
                    else:
                        G[u][v][kind_distance] = length
                        G[u][v][is_kind] = True if (length == G[u][v][weight]) else False
            i += 1

    return G


def distance_closure_selfloops(D, kind='metric', minkowski_par=1.0, algorithm='dijkstra', weight='weight', verbose=False, *args, **kwargs):
    """Computes the transitive closure (All-Pairs-Shortest-Paths; APSP)
    using different shortest path measures on the distance graph
    (adjacency matrix) with values in the ``[0,inf]`` interval.
    Including self-loops.

    .. math::

        c_{ij} = min_{k}( metric ( a_{ik} , b_{kj} ) )

    Parameters
    ----------
    D : NetworkX.Graph
        The Distance graph.

    kind : string
        Type of closure to compute: ``metric`` or ``ultrametric``.

    minkowski_par : float > 0
        Parameter of Minkowski distance

    algorithm : string
        Type of algorithm to use: ``dense`` or ``dijkstra``.

    weight : string
        Edge property containing distance values. Defaults to `weight`.
        
    Verbose :bool
        Prints statements as it computes.

    Returns
    --------
    C : NetworkX.Graph
        The distance closure graph. Note this may be a fully connected graph.

    See also: distance_closure
    """

    G = distance_closure(D, kind=kind, minkowski_par=minkowski_par, algorithm=algorithm, weight=weight, verbose=verbose)

    kind_distance = '{kind:s}_distance'.format(kind=kind)
    is_kind = 'is_{kind:s}'.format(kind=kind)

    # Self-loops
    for u, s in nx.selfloop_edges(G):
        length = G[u][u][weightt]
        for v in G.neighbors(u):
            if v != u and G.has_edge(v, u):
                new_length = G[u][v][kind_distance] + G[v][u][kind_distance]
                if new_length < length:
                    length = new_length
        G[u][u][kind_distance] = length
        G[u][u][is_kind] = True if (length == G[u][u][weight]) else False

    return G
