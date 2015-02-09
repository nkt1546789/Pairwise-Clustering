from pylab import *

def make_olympic_circles(n=200,e=0.01):
    cs=[[-1.5,0],[0,0],[1.5,0],[-0.75,-1],[0.75,-1]]
    X=[]
    y=[]
    for i,c in enumerate(cs):
        z=linspace(-1,1,n)
        X.append(c_[cos(2*pi*z)+c[0]+e*normal(size=n),
                    sin(2*pi*z)+c[1]+e*normal(size=n)])
        y.append(repeat(i,n))
    X=concatenate(X)
    y=concatenate(y)
    #print X.shape
    #scatter(X[:,0],X[:,1],c=y)
    #show()
    return X,y
