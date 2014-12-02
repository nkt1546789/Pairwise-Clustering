from pylab import *
from sklearn.metrics import pairwise_kernels,adjusted_rand_score
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV

class CCPM(BaseEstimator):
    def __init__(self,sigma=0.3,lam=1.,xce=None):
        self.sigma=sigma
        self.lam=lam
        self.xce=xce

    def fit(self,x):
        n=len(x)
        if self.xce is None:
            self.b=min(100,n)
            self.xce=x[permutation(n)][:self.b]
        else:
            self.b=len(self.xce)
        Phi=pairwise_kernels(x,self.xce,metric="rbf",gamma=1./(2*self.sigma**2))

        Phi1=tile(Phi.sum(0),(n,1))
        tmp1=Phi1.T.dot(Phi)/(n**2)
        tmp2=Phi.sum(0)/(n)
        self.alpha=pinv(tmp1 + self.lam*identity(self.b)).dot(tmp2)

        ppred1=maximum(Phi.dot(self.alpha),0.)
        ypred=ppred1>=0.5

        self.label=ypred
        return self

    def score(self,x):
        n=len(x)
        Phi=pairwise_kernels(x,self.xce,metric="rbf",gamma=1./(2*self.sigma**2))
        Phi1=tile(Phi.sum(0),(n,1))
        tmp1=Phi1.T.dot(Phi)/(n**2)
        tmp2=Phi.sum(0)/(n)
        score=self.alpha.dot(tmp1).dot(self.alpha)-self.alpha.dot(tmp2)
        return -score

class CCPMCV(CCPM):
    def fit(self,x):
        params={"sigma":logspace(-1,1,10),"lam":logspace(-1,1,10)}
        self=GridSearchCV(CCPM(),params).fit(x).best_estimator_
        return self

def main():
    n=500
    x,y=datasets.make_circles(n_samples=n,factor=.5,noise=.05)
    label=CCPMCV().fit(x).label
    print "ARI:",adjusted_rand_score(y,label)
    figure(1)
    scatter(x[:,0],x[:,1],c=label,s=50)
    show()

if __name__ == "__main__":
    main()
