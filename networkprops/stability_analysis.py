from __future__ import print
from networkprops import *
import networkx as nx
from numpy import *
import sys
import scipy.sparse as sprs
from numpy.random import normal


class stability_analysis(object):

    def __init__(self.A,sigma,mixing_gauss_delta=1.):

        if mixing_gauss_delta!=1.:
            print("mixing not yet implemented!")
            sys.exit(1)

        self.A = A.tolil()
        self.N = self.A.shape[0]
        self.node_indices = arange(self.N)
        self.maxiter = self.N*100

        self.sigma = sigma

    def fill_jacobian(self)

        row,col = self.A.nonzero()

        data = normal(scale=self.sigma,size=len(row)) 

        J = sprs.csr_matrix((data,(row,col)),dtype=float).tolil()
        J[self.node_indices,self.node_indices] = -1.

        self.J = J.tocsr()

        del J

        self.j_max = None

    def get_largest_real_eigenvalue(self,maxiter=-1):

        if self.j_max is None:

            if maxiter<=0:
                maxiter = self.maxiter

            j_large,_ = sprs.linalg.eigsh(self.J,k=2,which='LR',maxiter=maxiter)
            j_max = max(real(j_large))

        return self.j_max
