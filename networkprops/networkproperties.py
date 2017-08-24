import networkx as nx
import scipy.sparse as sprs
from numpy import *
import numpy as np
import time
import effdist
from networkprops import stability_analysis
import sys
from collections import Counter
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

class networkprops(object):

    def __init__(self,G,to_calculate=[],use_giant_component=False,weight='weight',catch_convergence_error=False,maxiter=-1,relabel_giant_component_to_integers=True):

        self.catch_convergence_error = catch_convergence_error
        G_basic = G

        self.NMAX = G_basic.number_of_nodes()

        if use_giant_component:
            subgraphs = nx.connected_component_subgraphs(G_basic,copy=False)
            self.G = max(subgraphs, key=len)
            if relabel_giant_component_to_integers:
                self.G = nx.convert_node_labels_to_integers(self.G)
        else:
            self.G = G_basic

        self.N = self.G.number_of_nodes()
        self.m = self.G.number_of_edges()

        self.weight = weight

        if maxiter <= 0:
            self.maxiter = self.N*100
        else:
            self.maxiter = maxiter

        self.sigma_for_eigs = 1e-10

        #self.get_eigenratio()        
        self.mean_degree = mean(np.array([d[1] for d in G.degree()],dtype=float))


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

    def mean_over_neighbors(self,func):
        """ func must be a function of the node """
        obs = 0.
        for n in G.nodes():
            kn = G.degree(n)
            for u in G.neighbors(n):
                if u != n:
                    obs += func(u) / float(kn)
        return obs

    def mean_over_unique_second_neighbors(self,func):
        """ func must be a function of the node """
        obs = 0.
        for n in G.nodes():
            first_neighbors = set(G.neighbors(n)) + { n }
            for u in first_neighbors:
                if u != n:
                    for v in G.neighbors(u):
                        if v not in first_neighbors:
                            #obs += 
                            pass

    #def mean_over_non_unique_second_neighbors(self,func):





    def get_laplacian(self):
        if not hasattr(self,"laplacian"):
            self.laplacian = sprs.csc_matrix(nx.laplacian_matrix(self.G,weight=self.weight),dtype=float)

        return self.laplacian

    def get_adjacency_matrix(self):
        if not hasattr(self,"adjacency_matrix") or self.adjacency_matrix is None:
            self.adjacency_matrix = sprs.csc_matrix(nx.adjacency_matrix(self.G,weight=self.weight),dtype=float)

        return self.adjacency_matrix

    def get_effective_distance(self):
        P = effdist.get_probability_graph(self.G,for_effdist=True)
        res = effdist.get_mean_effdist(P,with_error=False,get_all=True)

        return res, self.get_mean_and_err(res)

    def get_mean_and_err(self,arr):
        return mean(arr), std(arr)/sqrt(len(arr)-1.)


    def get_smallest_laplacian_eigenvalue(self,maxiter=-1):
        
        if not hasattr(self,"lambda_2") or self.lambda_2 is None:
            if not hasattr(self,"laplacian") or self.laplacian is None:
                self.get_laplacian()

            if maxiter<=0:
                maxiter = self.maxiter

            if self.catch_convergence_error:
                try:
                    lambda_small,_ = sprs.linalg.eigsh(self.laplacian,k=2,sigma=self.sigma_for_eigs,which='LM',maxiter=maxiter)
                except ArpackNoConvergence as e:
                    return None
            else:
                lambda_small,_ = sprs.linalg.eigsh(self.laplacian,k=2,sigma=self.sigma_for_eigs,which='LM',maxiter=maxiter)



            ind_zero = argmin(abs(lambda_small))
            lambda_small2 = delete(lambda_small,ind_zero)
            lambda_2 = min(lambda_small2)

            self.lambda_2 = lambda_2
            return lambda_2
        else:
            return self.lambda_2

    def get_n_smallest_laplacian_eigenvalues(self,n,maxiter=-1):

        if not hasattr(self,"lambda_smallest") or self.lambda_smallest is None:
            if not hasattr(self,"laplacian") or self.laplacian is None:
                self.get_laplacian()

            if maxiter<=0:
                maxiter = self.maxiter

            if self.catch_convergence_error:
                try:
                    lambda_small,_ = sprs.linalg.eigsh(self.laplacian,k=n+1,sigma=self.sigma_for_eigs,which='LM',maxiter=maxiter)
                except ArpackNoConvergence as e:
                    return None
            else:
                lambda_small,_ = sprs.linalg.eigsh(self.laplacian,k=n+1,sigma=self.sigma_for_eigs,which='LM',maxiter=maxiter)

            ind_zero = argmin(abs(lambda_small))
            lambda_small2 = delete(lambda_small,ind_zero)
            self.lambda_2 = min(lambda_small2)
            self.lambda_smallest = lambda_small2

            return self.lambda_smallest
        else:
            return self.lambda_smallest

    def get_largest_laplacian_eigenvalue(self,maxiter=-1):
        
        if not hasattr(self,"lambda_max") or self.lambda_max is None:
            if not hasattr(self,"laplacian") or self.laplacian is None:
                self.get_laplacian()

            if maxiter<=0:
                maxiter = self.maxiter

            lambda_large,_ = sprs.linalg.eigsh(self.laplacian,k=2,which='LM',maxiter=maxiter)
            lambda_max = max(real(lambda_large))

            self.lambda_max = lambda_max
            return lambda_max
        else:
            return self.lambda_max

    def get_eigenratio(self):

        if not hasattr(self,"eigenratio") or self.eigenratio is None:
            if not hasattr(self,"lambda_2") or self.lambda_2 is None:
                self.get_smallest_laplacian_eigenvalue()

            if not hasattr(self,"lambda_max") or self.lambda_max is None:
                self.get_largest_laplacian_eigenvalue()

            self.eigenratio = self.lambda_max/self.lambda_2

        return self.eigenratio

    def get_laplacian_eigenvalues(self,with_eigenvectors=False):

        if not hasattr(self,"laplacian_eigenvalues") or self.laplacian_eigenvalues is None:
            L = self.get_laplacian().toarray()
            self.laplacian_eigenvalues, self.laplacian_eigenvectors = linalg.eigh(L)

        if with_eigenvectors:
            return self.laplacian_eigenvalues, self.laplacian_eigenvectors
        else:
            return self.laplacian_eigenvalues

    def get_laplacian_eigenvalue_distribution(self,bins=20):

        if not hasattr(self,"laplacian_eigenvalues") or self.laplacian_eigenvalues is not None:
            lambdas = self.get_laplacian_eigenvalues()

        return histogram(lambdas,bins=bins,normed=True)

    def get_mean_first_passage_times_inverse_method(self,target,L=None,A=None,k=None):

        assert target < self.N
        assert target >= 0

        if L is None:
            L = self.get_laplacian()
        if A is None:
            A = self.get_adjacency_matrix()
        if k is None:
            k = np.array(A.sum(axis=1)).ravel()

        relevant_indices = np.concatenate(( np.arange(target), np.arange(target+1,self.N) ))

        L_prime = L[relevant_indices,:]
        L_prime = L_prime[:,relevant_indices]

        L_prime_inv = sprs.linalg.inv(L_prime)

        #Lambda = sprs.lil_matrix((self.N,self.N))

        #Lambda[:target,:target] = L_prime_inv[:target]

        tau = np.zeros((self.N,))

        #print L_prime_inv.T[target:,target:].shape, k[target+1:].shape
        #print L_prime_inv.T[:target,:target].shape, k[:target].shape

        if target == 0:
            tau[target+1:] += np.array(L_prime_inv.T[target:,target:].dot(k[target+1:])).ravel()
        elif target == self.N-1:
            tau[:target]  = np.array(L_prime_inv.T[:target,:target].dot(k[:target])).ravel()
        else:
            tau[:target]  = np.array(L_prime_inv.T[:target,:target].dot(k[:target])).ravel()
            tau[:target] += np.array(L_prime_inv.T[:target,target:].dot(k[target+1:])).ravel()
            tau[target+1:]  = np.array(L_prime_inv.T[target:,:target].dot(k[:target])).ravel()
            tau[target+1:] += np.array(L_prime_inv.T[target:,target:].dot(k[target+1:])).ravel()

        #for source in xrange(target):
        #    tau[source] = L_prime_inv.T.dot(k)

        return tau

    def get_mean_first_passage_times_for_all_targets_eigenvalue_method(self):
        k = np.array(self.get_adjacency_matrix().sum(axis=1)).ravel()
        lambdas, mus = self.get_laplacian_eigenvalues(with_eigenvectors=True)
        lambdas = lambdas[1:]

        # put eigenvectors as row vectors and disregard the first one
        # (corresponding to eigenvalue 0)
        mu = mus.T[1:,:]
        lambda_inv = 1./lambdas

        # Eq. (14) from https://arxiv.org/pdf/1209.6165v1.pdf
        T = self.N/(self.N-1.) * lambda_inv.dot( 2*self.m * mu**2 - mu*( mu.dot(k)[:,None] ) )

        return T

    def get_effective_resistance(self,source,target,mu=None,lambda_inv=None):

        if mu is None and lambda_inv is None:

            # get eigenvalues and eigenvectors
            lambdas, mus = self.get_laplacian_eigenvalues(with_eigenvectors=True)
            lambdas = lambdas[1:]

            # put eigenvectors as row vectors and disregard the first one
            # (corresponding to eigenvalue 0)
            mu = mus.T[1:,:]
            lambda_inv = 1./lambdas

        R = lambda_inv.dot( (mu[:,source]-mu[:,target])**2 )

        return R 

    def get_mean_effective_resistance(self,build_mean_over_two_point_values=False):

        # get eigenvalues and eigenvectors
        lambdas, mus = self.get_laplacian_eigenvalues(with_eigenvectors=True)
        lambdas = lambdas[1:]

        # put eigenvectors as row vectors and disregard the first one
        # (corresponding to eigenvalue 0)
        mu = mus.T[1:,:]
        lambda_inv = 1./lambdas

        if build_mean_over_two_point_values:
            R = 0.
            for source in xrange(self.N-1):
                for target in xrange(source+1,self.N):
                    if source != target:
                        R += self.get_effective_resistance(source,target,lambda_inv=lambda_inv,mu=mu)

            return R / self.N / (self.N-1.) 
        else:
            return lambda_inv.sum() / (self.N-1.)

        
    def get_mean_mean_first_passage_time(self,use_inverse_method=False):

        if use_inverse_method:
            L = self.get_laplacian()
            A = self.get_adjacency_matrix()
            k = np.array(A.sum(axis=1)).ravel()

            mean_tau = 0.

            for target in xrange(self.N): 
                tau = self.get_mean_first_passage_times_inverse_method(target,L,A,k)

                mean_tau += tau.sum()

            # norm by number of double counted pairs
            mean_tau /= self.N*(self.N-1)

            # This is a mean over all pairs. However, every pair has been counted twice
            return mean_tau
        else:
            # This is a mean over all pairs. However, every pair has been counted twice
            return np.mean(self.get_mean_first_passage_times_for_all_targets_eigenvalue_method())

    def get_eigenvalues(self,with_eigenvectors=False):

        if not hasattr(self,"eigenvalues") or self.eigenvalues is not None:
            A = self.get_adjacency_matrix().toarray()
            self.eigenvalues, self.eigenvectors = linalg.eigh(A)

        if with_eigenvectors:
            return self.eigenvalues, self.eigenvectors
        else:
            return self.eigenvalues

    def get_eigenvalue_distribution(self,bins=20):

        if not hasattr(self,"eigenvalues") or self.eigenvalues is not None:
            alphas = self.get_eigenvalues()

        return histogram(alphas,bins=bins,normed=True)


    def get_largest_eigenvalue(self,maxiter=-1):

        if not hasattr(self,"a_max") or self.a_max is None:
            if not hasattr(self,"adjacency_matrix") or self.adjacency_matrix is None:
                self.get_adjacency_matrix()

            if maxiter<=0:
                maxiter = self.maxiter

            a_large,_ = sprs.linalg.eigsh(self.adjacency_matrix,k=2,which='LM',maxiter=maxiter)
            a_max = max(real(a_large))

            self.a_max = a_max

            return a_max
        else:
            return self.a_max

    def get_path_lengths(self):
        if not hasattr(self,"shortest_path_lenghts") or self.shortest_path_lenghts is None:
            self.shortest_paths_lengths = nx.all_pairs_shortest_path_length(self.G)
            self.avg_shortest_path = sum([ length for sp in self.shortest_paths_lengths.values() for length in sp.values() ])/float(self.N*(self.N-1))
            self.eccentricity = nx.eccentricity(self.G,sp=self.shortest_paths_lengths)
            self.diameter = nx.diameter(self.G,e=self.eccentricity)
            self.radius = nx.radius(self.G,e=self.eccentricity)

        return self.shortest_paths_lengths

    def get_avg_shortest_path(self):
        if not hasattr(self,"shortest_path_lenghts") or self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.avg_shortest_path
        
    def get_eccentricity(self):
        if not hasattr(self,"shortest_path_lenghts") or self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.eccentricity
        
    def get_diameter(self):
        if not hasattr(self,"shortest_path_lenghts") or self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.diameter
        
    def get_radius(self):
        if not hasattr(self,"shortest_path_lenghts") or self.shortest_path_lenghts is None:
            self.get_path_lengths()

        return self.radius

    def get_betweenness_centrality(self):
        
        if not hasattr(self, "betweenness_centrality") or self.betweenness_centrality is None:
            self.betweenness_centrality = nx.betweenness_centrality(self.G)
            self.max_B = max(self.betweenness_centrality.values())
            self.min_B = min(self.betweenness_centrality.values())
            self.mean_B,self.mean_B_err = self.get_mean_and_err(self.betweenness_centrality.values())

        return self.betweenness_centrality, self.mean_B,self.mean_B_err

    def get_min_max_betweenness_centrality(self):

        if not hasattr(self, "betweenness_centrality") or self.betweenness_centrality is None:
            self.get_betweenness_centrality()

        return self.min_B, self.max_B

    def stability_analysis(self,sigma,N_measurements=1,mode="random",maxiter=None,tol=-1.):
        
        j_max = zeros(N_measurements)

        for meas in range(N_measurements):
            if not hasattr(self,"adjacency_matrix") or self.adjacency_matrix is None:
                self.get_adjacency_matrix()

            stab_ana = stability_analysis(self.get_adjacency_matrix(),sigma,maxiter=maxiter,tol=tol)

            if mode=="random":
                stab_ana.fill_jacobian_random()
            elif mode=="predatorprey":
                stab_ana.fill_jacobian_predator_prey()
            elif mode=="mutualistic":
                stab_ana.fill_jacobian_mutualistic()
            else: 
                print "Mode",mode,"not known."
                sys.exit(1)

            j_max[meas] = stab_ana.get_largest_realpart_eigenvalue()

        if N_measurements>1:
            return self.get_mean_and_err(j_max)
        else:
            return j_max[0]

    def get_degree_distribution(self,k_min=0):
        degrees = self.G.degree().values()
        dist = Counter(degrees)
        k_min = 0
        k_max = max(degrees)
        ks = arange(k_min,k_max+1)
        vals = array([dist[k] for k in ks])

        return ks,vals

    def get_clustering_by_degree(self,get_mean_by_degree=False):
        deg = self.G.degree().values()
        C_ = nx.clustering(self.G)
        k = zeros((self.N,))
        C = zeros((self.N,))
        for i,node in enumerate(self.G.nodes()):
            k[i] = deg[node]
            C[i] = C_[node]

        return k, C

    def get_mean_local_clustering(self):
        C = nx.clustering(self.G)
        return np.mean(C.values())


if __name__=="__main__":
    import mhrn
    import pylab as pl
    import seaborn as sns
    from nwDiff import ErgodicDiffusion

    test_stability = False
    test_mean_tau = True

    G = mhrn.fast_mhr_graph(B=10,L=2,k=7,xi=1.4)

    nprops = networkprops(G,use_giant_component=True)
    neigh,mea_err = nprops.get_unique_second_neighbors()
    effdist,mea_err = nprops.get_effective_distance()
    lam_2 = nprops.get_smallest_laplacian_eigenvalue()
    lam_m = nprops.get_largest_laplacian_eigenvalue()
    eigratio = nprops.get_eigenratio()
    Bstuff = nprops.get_betweenness_centrality()
    Bmin,Bmax = nprops.get_min_max_betweenness_centrality()

    path_len = nprops.get_path_lengths()
    sp = nprops.get_avg_shortest_path()
    
    e = nprops.get_eccentricity()
    d = nprops.get_diameter()
    r = nprops.get_radius()

    alpha = nprops.get_largest_eigenvalue()

    vals,bars = nprops.get_laplacian_eigenvalue_distribution()
    pl.step(bars[:-1],vals)

    if test_stability:

        jmax,jerr = nprops.stability_analysis(0.15,10,tol=1e-2)
        print jmax, jerr

        jmax,jerr = nprops.stability_analysis(0.15,10,mode="mutualistic",tol=1e-2)
        print jmax, jerr

        jmax,jerr = nprops.stability_analysis(0.15,10,mode="predatorprey",maxiter=20000,tol=1e-2)
        print jmax, jerr

    if test_mean_tau:
        r, h = 3, 3 
        G = nx.balanced_tree(r,h)
        props = networkprops(G)

        tau = props.get_mean_mean_first_passage_time(use_inverse_method=True)
        print "inverse laplacian method mean tau =", tau

        tau = props.get_mean_effective_resistance()
        print "tree laplacian eigenvalue method mean tau =", tau

        tau = props.get_mean_mean_first_passage_time()
        print "general laplacian eigenvalue method mean tau =", tau

        diff = ErgodicDiffusion(G)

        N_meas = 10
        tau = 0.
        for meas in xrange(N_meas):
            MFPT, coverage_time = diff.simulate_for_MFPT_and_coverage_time()
            #fig,ax = pl.subplots(1,1)
            #ax.hist(MFPT.flatten(),bins=np.arange(min(MFPT.flatten()), max(MFPT.flatten()) + 5, 5))
            tau += diff.get_mean_MFPT()
        tau /= N_meas

        print "simulated mean tau =", tau

        print "mean effective resistance (over two point) =", props.get_mean_effective_resistance(build_mean_over_two_point_values=True)


    pl.show()



