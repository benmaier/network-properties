from numpy import *
import networkx as nx
#from networkx.linalg.spectrum import laplacian
from config_file import *
import scipy.sparse.linalg as scla
import scipy.sparse as sprs
import scipy.linalg as scla_std
import sys
import getopt
import string
import os
import time
import community
from effdist import *

def error(message):
    sys.stderr.write("error: %s\n" % message)
    sys.exit(1)

#=====================================================
argv = sys.argv[1:]
helpstring = 'python properties.py -p <resultpath> -g <graph_id> -k <degree_id> -s <seed> -v [if verbose] --scalefree [if fill scalefree] --gamma <value for scalefree choosing function>'

ik = 0
g = 0
resultpath = "./results"
verbose = False
scalefree = False
gamma = 1.0

try:
    opts, args = getopt.getopt(argv,"hvg:p:k:s:",["scalefree","gamma="])
except getopt.GetoptError:
    print helpstring
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print helpstring
        sys.exit()
    elif opt=="-g":
        g = int(arg)
    elif opt=="-k":
        ik = int(arg)
    elif opt=="-p":
        resultpath = arg
    elif opt=="-v":
        verbose = True
    elif opt=="--scalefree":
        scalefree = True
    elif opt=="--gamma":
        gamma = float(arg)
    elif opt=="-s":
        random.seed(int(arg))


obs = zeros((N_observables,
             len(Ls),
             len(xis) 
            ));

if verbose:
    print "obs has the following size:", size(obs)


k = ks[ik]
filename = resultpath + "/k_" + str(k) + "__g_" + str(g)
if verbose:
    print resultpath
cum_start = time.time()

sigma_for_eigs = 1e-10

for L in Ls:
    iL = list(Ls).index(L)

    for xi in xis:
        ixi = list(xis).index(xi)

        #get name of the network for documentation of what the program is doing
        if verbose:
            print "Now generating: L =", L,",  k =",  k,",  xi =",xi

        start = time.time()
        #generate graph and find the giant component
        G_basic = fast_mhr_graph(B,L,k,xi,scalefree=scalefree,gamma=gamma)
        NMAX = float(B**L)
        subgraphs = nx.connected_component_subgraphs(G_basic,copy=False)
        G = max(subgraphs, key=len)
        N = G.number_of_nodes()
        m = G.number_of_edges()

        end = time.time()
        if verbose:
            print "time needed for generation:", end-start,"s"

        start = time.time()
        A_ = sprs.csc_matrix(nx.adjacency_matrix(G),dtype=float)

        #compute largest eigenvalues
        start = time.time()
        a_large,_ = sprs.linalg.eigsh(A_,k=2,which='LM',maxiter=B**L*50)
        end = time.time()
        if verbose:
            print "time needed for largest eigenvalue: ", end-start,"s"

        a_max = max(real(a_large))

        end = time.time()
        if verbose:
        #    print "time needed for eigenvalues:", end-start,"s"
             print "a_max =", a_max
             print " "


        #Laplacian
        L_ = sprs.csc_matrix(nx.laplacian_matrix(G),dtype=float)
        if L>=2 and xi>.4:
            #sigma_for_eigs = obs[LAMBTWO][iL][ixi-2]/1.5
            #print sigma_for_eigs

            start = time.time()
            lambda_small,_ = sprs.linalg.eigsh(L_,k=2,which='SM',maxiter=B**L*50)
            end = time.time()
            if verbose:
                print "time needed for smallest eigenvalues:", end-start,"s"
        else:
            start = time.time()
            lambda_small,_ = sprs.linalg.eigsh(L_,k=2,sigma=sigma_for_eigs,which='LM',maxiter=B**L*50)
            end = time.time()
            if verbose:
                print "time needed for smallest eigenvalues:", end-start,"s"
            

        ind_zero = argmin(abs(lambda_small))
        lambda_small2 = delete(lambda_small,ind_zero)
        lambda_2 = min(lambda_small2)

        #compute largest eigenvalues
        start = time.time()
        lambda_large,_ = sprs.linalg.eigsh(L_,k=2,which='LM',maxiter=B**L*50)
        end = time.time()
        if verbose:
            print "time needed for largest eigenvalues: ", end-start,"s"
            #print " "

        lambda_max = max(lambda_large)


        #compute modularity
        start = time.time()
        partition = community.best_partition(G)
        modularity = community.modularity(partition,G)
        end = time.time()
        if verbose:
             print "time needed for modularity: ", end-start,"s"
             print "modularity =", modularity
             print " "

        #compute shortest path lengths
        start = time.time()
        shortest_paths_lengths = nx.all_pairs_shortest_path_length(G)
        avg_shortest_path = sum([ length for sp in shortest_paths_lengths.values() for length in sp.values() ])/float(N*(N-1))
        eccentricity = nx.eccentricity(G,sp=shortest_paths_lengths)
        diameter = nx.diameter(G,e=eccentricity)
        radius = nx.radius(G,e=eccentricity)
        end = time.time()
        if verbose:
             print "time needed for all shortest paths: ", end-start,"s"
             print "avg_shortest_path =", avg_shortest_path
             print "avg_eccentricity =", mean(eccentricity.values())
             print "diameter =", diameter
             print "radius =", radius
             print " "

        #compute betweenness centrality
        start = time.time()
        betweenness_centrality = nx.betweenness_centrality(G)
        mean_B = mean(betweenness_centrality.values()) 
        max_B = max(betweenness_centrality.values())
        min_B = min(betweenness_centrality.values())
        end = time.time()
        if verbose:
             print "time needed for betweenness centrality: ", end-start,"s"
             print "mean_betweenness =", mean_B
             print "max_betweenness =", max_B
             print "min_betweenness =", min_B
             print " "

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
        sec_neigh = mean(num_of_second_neighs)
        end = time.time()
        if verbose:
             print "time needed for second neighbors: ", end-start,"s"
             print "number of unique second neighbors =", sec_neigh
             print " "

        #compute mean effective distance
        start = time.time()
        P = get_probability_graph(G,for_effdist=True)
        effdist = get_mean_effdist(P)
        end = time.time()
        if verbose:
             print "time needed for effective distance: ", end-start,"s"
             print "mean effdist =", effdist
             print " "

        #save results
        obs[GIA][iL][ixi] = N/NMAX
        obs[MOD][iL][ixi] = modularity
        obs[SHO][iL][ixi] = avg_shortest_path
        obs[ECC][iL][ixi] = mean(eccentricity.values())
        obs[DIA][iL][ixi] = diameter
        obs[RAD][iL][ixi] = radius
        obs[BET][iL][ixi] = mean_B
        obs[B_X][iL][ixi] = max_B
        obs[B_N][iL][ixi] = min_B
        obs[A_X][iL][ixi] = a_max
        obs[L_X][iL][ixi] = lambda_max
        obs[L_2][iL][ixi] = lambda_2
        obs[EIG][iL][ixi] = lambda_max/lambda_2
        obs[SEC][iL][ixi] = sec_neigh
        obs[EFF][iL][ixi] = effdist

        #clear memory
        G_basic.clear()
        del G_basic
        del partition
        del shortest_paths_lengths
        del num_of_second_neighs
        del first_neighs
        del second_neighs
        del eccentricity
        for g in subgraphs:
            g.clear()
        #del g[:]
        G.clear()
        del G
        del P
        del A_
        del L_

cum_end = time.time()

if verbose:
    print "\n cumulated time needed:", cum_end-cum_start,"s"

save(filename,obs)
