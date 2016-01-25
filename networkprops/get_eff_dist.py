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
helpstring = 'python get_eff_dist.py -p <resultpath> -g <graph_id> -k <degree_id> -s <seed> -v [if verbose]'

ik = 0
g = 0
resultpath = "./results"
verbose = False

try:
    opts, args = getopt.getopt(argv,"hvg:p:k:s:")
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
        G_basic = fast_mhr_graph(B,L,k,xi)
        NMAX = float(B**L)
        subgraphs = nx.connected_component_subgraphs(G_basic,copy=False)
        G = max(subgraphs, key=len)
        N = G.number_of_nodes()
        m = G.number_of_edges()


        end = time.time()
        if verbose:
            print "time needed for generation:", end-start,"s"

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
        obs[0][iL][ixi] = effdist

        #clear memory
        G_basic.clear()
        del G_basic
        for g in subgraphs:
            g.clear()
        #del g[:]
        G.clear()
        del G
        del P

cum_end = time.time()

if verbose:
    print "\n cumulated time needed:", cum_end-cum_start,"s"

save(filename,obs)
