import numpy as np
from networkprops import networkprops as nprops

import networkx as nx

N = 200
k = 3
p = k / (N-1)

#G = nx.barabasi_albert_graph(N,k)
G = nx.grid_2d_graph(int(np.sqrt(N)),int(np.sqrt(N)))

prop = nprops(G, use_giant_component = True)

T1 = prop.get_mean_first_passage_times_for_all_targets_eigenvalue_method()
T2 = prop.get_global_mean_first_passage_times_from_stationary_state(P_stationary = np.ones(prop.N)/prop.N)

import matplotlib.pyplot as pl

pl.figure()

pl.plot(T1)
pl.plot(T2)

T3 = prop.get_mean_first_passage_times_for_all_targets_eigenvalue_method(use_stationary_distribution=True)
T4 = prop.get_global_mean_first_passage_times_from_stationary_state()

pl.figure()

pl.plot(T1)
pl.plot(T3)
pl.plot(T4)

print(T1.mean(),T4.mean())


pl.show()
