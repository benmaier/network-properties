import numpy as np
from networkprops import networkprops as nprops

import networkx as nx

N = 100
k = 10
p = k / (N-1)

G = nx.fast_gnp_random_graph(N,p)

prop = nprops(G, use_giant_component = True)

T1 = prop.get_mean_first_passage_times_for_all_targets_eigenvalue_method()
T2 = prop.get_global_mean_first_passage_times_from_stationary_state(P_stationary = np.ones(N)/(N-1))

import matplotlib.pyplot as pl

pl.plot(T1)
pl.plot(T2)
pl.show()
