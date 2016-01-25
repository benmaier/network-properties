import networkx as nx
import scipy.sparse as sprs
from numpy import *
import time
import effdist

class networkprops(object):

    def __init__(G,to_calculate=[],use_giant_component=False,weight='weight'):

        G_basic = G

        self.NMAX = G_basix.number_of_nodes()

        if use_giant_component:
            subgraphs = nx.connected_component_subgraphs(G_basic,copy=False)
            self.G = max(subgraphs, key=len)
        else:
            self.G = G_basic

        self.N = G.number_of_nodes()
        self.m = G.number_of_edges()

        self.weight = weight

        self.maxiter = self.N*100


    def calculate_all(self):
        pass


    def get_all(self):
        pass

    def get_unique_second_neighbors(self):
        #compute number of unique second neighbors
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

        res = array(num_of_second_neighs)

        return res, self.get_mean_and_err(res)

    def get_effective_distance(self):
        P = effdist.get_probability_graph(self.G,for_effdist=True)
        res = effdist.get_mean_effdist(self.G,with_error=True,get_all=True)

        return res, self.get_mean_and_err(res)

    def get_mean_and_err(self,arr):
        return mean(arr), std(arr)/sqrt(len(arr)-1.)


    def get_smallest_laplacian_eigenvalue(self,maxiter=-1):
        
        if self.lambda_2 is None:
            if self.laplacian is None:
                self.laplacian = nx.laplacian_matrix(self.G,weight=weight)

            if maxiter<=0:
                maxiter = self.maxiter

            lambda_small,_ = sprs.linalg.eigsh(L_,k=2,sigma=sigma_for_eigs,which='LM',maxiter=maxiter)

            ind_zero = argmin(abs(lambda_small))
            lambda_small2 = delete(lambda_small,ind_zero)
            lambda_2 = min(lambda_small2)

            self.lambda_2 = lambda_2
            return lambda_2
        else:
            return self.lambda_2

    def get_largest_laplacian_eigenvalue(self,maxiter=-1):
        
        if self.lambda_max is None:
            if self.laplacian is None:
                self.laplacian = nx.laplacian_matrix(self.G,weight=weight)

            if maxiter<=0:
                maxiter = self.maxiter

            lambda_large,_ = sprs.linalg.eigsh(L_,k=2,which='LM',maxiter=maxiter)
            lambda_max = max(real(lambda_large))

            self.lambda_max = lambda_max
            return lambda_max
        else:
            return self.lambda_max

    def get_eigenratio()

        if self.eifenratio is None:
            if self.lambda_2 is None:
                self.get_smallest_laplacian_eigenvalue()

            if self.lambda_max is None:
                self.get_largest_laplacian_eigenvalue()

            self.eigenratio = self.lambda_max/self.lambda_2

        return self.eigenratio


    def get_largest_eigenvalue(self,maxiter=-1):

        if self.a_max is None:
            if self.adjacency_matrix is None:
                self.adjacency_matrix = nx.adjacency_matrix(self.G,weight=weight)

            if maxiter<=0:
                maxiter = self.maxiter

            a_large,_ = sprs.linalg.eigsh(A_,k=2,which='LM',maxiter=maxiter)
            a_max = max(real(a_large))

            self.a_max = a_max

            return a_max
        else:
            return self.a_max

    def get_path_lengths(self):
        if self.shortest_path_lenghts is None:
            self.shortest_paths_lengths = nx.all_pairs_shortest_path_length(self.G)
            self.avg_shortest_path = sum([ length for sp in shortest_paths_lengths.values() for length in sp.values() ])/float(N*(N-1))
            self.eccentricity = nx.eccentricity(self.G,sp=shortest_paths_lengths)
            self.diameter = nx.diameter(self.G,e=eccentricity)
            self.radius = nx.radius(self.G,e=eccentricity)

        return self.shortest_paths_lengths

    def get_avg_shortest_path(self):
        if self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.avg_shortest_path
        
    def get_eccentricity(self):
        if self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.eccentricity
        
    def get_diameter(self):
        if self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.diameter
        
    def get_radius(self):
        if self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.radius

    def get_betweenness_centrality(self):
        
        if self.betweenness_centrality is None:
            self.betweenness_centrality = nx.betweenness_centrality(self.G)
            self.mean_B = mean(self.betweenness_centrality.values()) 
            self.max_B = max(self.betweenness_centrality.values())
            self.min_B = min(self.betweenness_centrality.values())
            self.mean_B,self.mean_B_err = self.get_mean_and_error(self.betweenness_centrality)

        return self.betweenness_centrality, self.mean_B,self.mean_B_err

    def get_min_max_betweenness_centrality(self):

        if self.betweenness_centrality is None:
            self.get_betweenness_centrality()

        return self.min_B, self.max_B

        
