from __future__ import print_function
from networkprops import *
import networkx as nx
from numpy import *
import sys
import scipy.sparse as sprs
from numpy.random import normal


class stability_analysis(object):

    def __init__(self,A,sigma,self_interaction=1.,mixing_gauss_delta=1.,maxiter=None,tol=-1.):
        """
        Takes a network, computes
        """

        if not mixing_gauss_delta==1.:
            print("mixing not yet implemented!")
            sys.exit(1)

        if type(A) == tuple:
            N, row, col = A
            data = ones_like(row)
            self.A = sprs.coo_matrix((data,(row,col)), shape=(N,N))
        else:
            self.A = A.tocoo()

        self.N = self.A.shape[0]
        self.node_indices = arange(self.N)

        if maxiter is None:
            self.maxiter = self.N*1000
        else:
            self.maxiter = maxiter

        if tol<0.:
            self.tol = 0.
        else:
            self.tol = tol

        self.sigma = sigma

        #this goes onto the diagonal 
        self.self_interaction = self_interaction 

    def fill_jacobian_random(self):

        row,col = self.A.nonzero()

        data = normal(scale=self.sigma,size=len(row)) 

        J = sprs.csr_matrix((data,(row,col)),dtype=float).tolil()
        J[self.node_indices,self.node_indices] = -1.

        self.J = J.tocsr()

        del J

        self.j_max = None

    def fill_jacobian_predator_prey(self):

        row,col = self.A.nonzero()
        data = empty_like(row,dtype=float)

        upper_triangle = nonzero(row<col)[0]
        lower_triangle = nonzero(row>col)[0]

        data[upper_triangle] = abs(normal(scale=self.sigma,size=len(upper_triangle)))
        data[lower_triangle] = -abs(normal(scale=self.sigma,size=len(lower_triangle)))

        self.J = sprs.csc_matrix((data,(row,col)),dtype=float) - self.self_interaction * sprs.eye(self.N)
        

        self.j_max = None

    def fill_jacobian_mutualistic(self):

        row,col = self.A.nonzero()
        data = zeros_like(row,dtype=float)

        upper_triangle = nonzero(row<col)[0]
        lower_triangle = nonzero(row>col)[0]

        data[upper_triangle] = normal(scale=self.sigma,size=len(upper_triangle))
        self.J = sprs.csr_matrix((data,(row,col)),dtype=float)

        signs = array([ sign(self.J[col[ndx],row[ndx]]) for ndx in lower_triangle ])
        data[lower_triangle] = signs * abs(normal(scale=self.sigma,size=len(lower_triangle)))

        self.J = sprs.csr_matrix((data,(row,col)),dtype=float) - self.self_interaction * sprs.eye(self.N,dtype=float)

        #for ndx in upper_triangle:
        #    print("===",self.J[row[ndx],col[ndx]], self.J[col[ndx],row[ndx]])

        self.j_max = None


    def fill_jacobian_proportions(self):
        pass

         



    def get_largest_realpart_eigenvalue(self,maxiter=-1,tol=-1.):

        if self.j_max is None:

            if maxiter<=0:
                maxiter = self.maxiter
            
            if tol<0.:
                tol = self.tol

            j_large = sprs.linalg.eigs(self.J,k=2,which='LR',maxiter=maxiter,tol=tol,return_eigenvectors=False)
            self.j_max = max(real(j_large))

        return self.j_max


if __name__ == "__main__":
    import cMHRN

    A = cMHRN.fast_mhrn_coord_lists(8,3,7,8,use_giant_component=True)

    stab = stability_analysis(A,sigma=2)
    stab.fill_jacobian_predator_prey()
    print(stab.get_largest_realpart_eigenvalue())
