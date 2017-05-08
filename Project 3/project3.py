import random
import time

import numpy as np, pandas as pd

from scipy.stats import multivariate_normal

def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an nxd ndarray
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    n, d = data.shape
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
    
    old_cost = 1+eps
    new_cost = 0
    cluster_assignments = np.zeros(n)
    # check if stop is met
    while abs(old_cost - new_cost) > eps:
        # assign data to clusters
        for i in range(n):
            low_dist = None
            for j in range(k):
                # based on distance from centroid
                dist = distance(mu[j,:], data[i,:], 2)
                if low_dist == None or dist <= low_dist:
                    low_dist = dist
                    cluster_assignments[i] = j
        # update centroids
        for j in range(k):
            c = 0
            centroid = np.zeros(d)
            for i in range(n):
                if cluster_assignments[i] == j:
                    c = c + 1
                    centroid = centroid + data[i,:]
                mu[j,:] = centroid/c
        # new cost from distance
        old_cost = new_cost
        new_cost = 0
        for j in range(k):
            for i in range(n):
                if cluster_assignments[i] == j:
                    new_cost = new_cost + distance(mu[j,:], data[i,:], 2)
    print(new_cost)
    return (mu, cluster_assignments)

# helper function for distance
def distance(A, B, num):
    s = 0
    for i in range(A.size):
        s = s + np.power(abs(A[i] - B[i]), num)
    return np.power(s, 1/num)

class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-4, verbose=True, max_iters=100):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True


class GMM(MixtureModel):
    def __init__(self, k, d):
        super(GMM, self).__init__(k)
        self.params['mu'] = np.random.randn(k, d)

    def e_step(self, data):
        n = data.shape[0]
        k = self.k
        
        L = 0
        p = np.zeros([n, k])
        for i in range(n):
            s = 0
            for j in range(k):
                s = s + self.pi[j]*multivariate_normal.pdf(data[i,:], self.mu[j,:], self.sigsq[j])
            for j in range(k):
                p[i, j] = self.pi[j]*multivariate_normal.pdf(data[i,:], self.mu[j,:], self.sigsq[j])/s
            L = L + np.log(s)
        return (L, p)

    def m_step(self, data, pz_x):
        n = data.shape[0]
        d = data.shape[1]
        k = self.k
        
        nj = np.zeros([k])
        for j in range(k):
            nj[j] = np.sum(pz_x[:,j])
        
        new_pi = nj/n
        
        new_mu = np.zeros([k, d])
        for j in range(k):
            for i in range(n):
                new_mu[j,:] = new_mu[j,:] + pz_x[i,j]*data[i,:]/nj[j]
            
        new_sigsq = np.zeros([k])
        for j in range(k):
            for i in range(n):
                new_sigsq[j] = new_sigsq[j] + pz_x[i,j]*np.power(distance(np.zeros(d), data[i,:] - new_mu[j,:], 2), 2)/(2*nj[j])

        return {
            'pi': new_pi,
            'mu': new_mu,
            'sigsq': new_sigsq,
        }

    def fit(self, data, *args, **kwargs):
        self.params['sigsq'] = np.asarray([np.mean(data.var(0))] * self.k)
        return super(GMM, self).fit(data, *args, **kwargs)


class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds]

    def e_step(self, data):
        n = data.shape[0]
        k = self.k
        d = len(self.alpha)
            
        p = np.ones([n, k])
        for t in range(d):
            dumx = pd.get_dummies(data.iloc[:,t], dummy_na=True)
            alp = np.append(self.alpha[t], np.ones([self.alpha[t].shape[0],1]), 1)
            p = p*np.dot(dumx,np.transpose(alp)) # n x np + 1 . np + 1 x k
        p = np.multiply(p,self.pi)
        for i in range(n):
            p[i,:] = p[i,:]/sum(p[i,:])
        
        L = sum(np.dot(p,np.log(self.pi))) # n x k . k x 1 Double Summation
        for t in range(d):
            dumx = pd.get_dummies(data.iloc[:,t], dummy_na=True)
            alp = np.append(self.alpha[t], np.ones([self.alpha[t].shape[0],1]), 1)
            p_xa = np.dot(dumx,np.transpose(alp)) # n x np + 1 . np + 1 x k
            L = L + np.sum(np.multiply(p,np.log(p_xa))) # Triple Summation
        return (L, p)

    def m_step(self, data, p_z):
        n = data.shape[0]
        k = self.k
        d = len(self.alpha)
        
        nj = np.zeros([k]) # k x 1
        for j in range(k):
            nj[j] = np.sum(p_z[:,j])
            
        new_pi = nj/n  # k x 1
        
        p_zt = np.transpose(p_z)
        new_alpha = []
        for t in range(d):
            dumx = pd.get_dummies(data.iloc[:,t])
            a_sum = np.dot(p_zt, dumx) # k x n . n x np = k x np
            for j in range(k):
                a_sum[j,:] = a_sum[j,:]/np.sum(a_sum[j,:])
            new_alpha.append(a_sum) # k x np / k x 1
            
        return {
            'pi': new_pi,
            'alpha': new_alpha,
        }

    @property
    def bic(self):
        d = len(self.alpha)
        
        p = 0
        for t in range(d):
            p = p + self.alpha[t].shape[0]*(self.alpha[t].shape[1] - 1) # shape k x np
            # alpha[t][k,:] must sum to 1 so only np-1 independent parameters
        return self.max_ll - 0.5*p*np.log(self.n_train)
