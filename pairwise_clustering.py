from pylab import *
from sklearn.metrics import pairwise_kernels,adjusted_rand_score
from sklearn import datasets
from sklearn.base import BaseEstimator
from itertools import combinations
from CCPM import CCPMCV
from data import make_olympic_circles

class PairwiseClustering(BaseEstimator):
    def fit(self,x):
        idx=array(list(combinations(range(len(x)),2)))
        xt=x[idx[:,0]]-x[idx[:,1]]
        ntr=min(len(xt),1000)
        xt_tr=xt[permutation(len(xt))][:ntr]
        z=CCPMCV().fit(xt_tr).predict(xt)
        if sum(z==1)>sum(z==0):
            t=0
        else:
            t=1
        self.label_=arange(len(x))
        for k,(i,j) in enumerate(idx):
            if z[k]==1:
                self.label_[i]=self.label_[j]
        for k,cls in enumerate(unique(self.label_)):
            self.label_[self.label_==cls]=k
        return self

def main():
    n=200
    x,y=datasets.make_blobs(n_samples=n, random_state=8)
    #x,y=make_olympic_circles(n=n,e=0.05)
    #x,y=datasets.make_circles(n_samples=n,factor=.5,noise=.05)
    #x,y=datasets.make_moons(n_samples=n,noise=.05)

    label=PairwiseClustering().fit(x).label_
    print "clusters:", len(unique(label))
    print "ARI:",adjusted_rand_score(y,label)
    #figure(1)
    #scatter(x[:,0],x[:,1],c=label,s=50)
    #show()

if __name__ == '__main__':
    main()
