from pylab import *
from sklearn.metrics import pairwise_kernels,adjusted_rand_score
from sklearn import datasets
from itertools import combinations
from CCPM import CCPMCV

class PairwiseClustering(object):
    def fit(self,x):
        comb=list(combinations(range(len(x)),2))
        xt=[]
        for i,j in comb:
            xt.append(x[i]-x[j])
        xt=array(xt)
        z=CCPMCV().fit(xt).label
        self.label_=arange(len(x))
        for k,(i,j) in enumerate(comb):
            if z[k]==1:
                self.label_[i]=self.label_[j]
        for k,cls in enumerate(unique(self.label_)):
            self.label_[self.label_==cls]=k
        return self

def main():
    n=300
    x,y=datasets.make_blobs(n_samples=n, random_state=8)
    label=PairwiseClustering().fit(x).label_
    print "ARI:",adjusted_rand_score(y,label)
    figure(1)
    scatter(x[:,0],x[:,1],c=label,s=50)
    show()

if __name__ == '__main__':
    main()
