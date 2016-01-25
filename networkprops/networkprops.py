import networkx as np
import scipy.sparse as sprs
from numpy import *
import time
import effdist

class networkprops:

    def __init__(G,to_calculate=[],use_giant_component=False):

        G_basic = G

        self.NMAX = G_basix.number_of_nodes()

        if use_giant_component:
            subgraphs = nx.connected_component_subgraphs(G_basic,copy=False)
            self.G = max(subgraphs, key=len)
        else:
            self.G = G_basic

        self.N = G.number_of_nodes()
        self.m = G.number_of_edges()


    def calculate_all():
        pass


    def get_all():
        pass

    def get_unique_second_neighbors():
        #compute number of unique second neighbors
        start = time.time()
        num_of_second_neighs = []
        for n in G.nodes():
            first_neighs = set(G.neighbors(n))
            first_neighs.add(n)
            second_neighs = set([])
            for neigh in first_neighs:
                second_neighs.update(set(G.neighbors(neigh)))
            second_neighs = second_neighs - first_neighs    
            num_of_second_neighs.append(len(second_neighs))

        end = time.time()
        if self.verbose:
             print "time needed for second neighbors: ", end-start,"s"
             print "number of unique second neighbors =", sec_neigh
             print " "

        res = array(num_of_second_neighs)

        return res, self.get_mean_and_err(res)

    def get_mean_and_err(self,arr):
        return mean(arr), std(arr)/sqrt(len(arr)-1.)

    def get_effective_distance():
        P = effdist.get_probability_graph(self.G,for_effdist=True)
        res = effdist.get_mean_effdist(self.G,with_error=True,get_all=True)

        return res, self.get_mean_and_err(res)

