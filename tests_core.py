import numpy as np, time
from rgmap.core import *
from rgmap import core
# Jacobian check via central differences in extended precision-free way: use small h on random states (no attractor), several configs
rng=np.random.default_rng(0)
def fd(Q,P,h=1e-6):
    B,N,_=Q.shape; n=3*N; J=np.zeros((B,n,n))
    for c in range(n):
        E=np.zeros(n); E[c]=h; E=E.reshape(N,3)
        J[:,:,c]=((terms(Q+E,P)[3]-terms(Q-E,P)[3]).reshape(B,-1))/(2*h)
    return J
for N,p,phi,fs,ms in [(3,3,True,1,(1,1,1)),(4,3,True,1,(1,2,.5,1)),(3,0,True,.3,(1,1,1)),(5,5,False,1,(1,)*5),(2,3,True,1,(1,1))]:
    P=Params(np.array([0.97,2.0]),eps=np.array([0.1,0.2]),masses=ms,p=p,use_phi=phi,force_scale=fs)
    Q=rng.uniform(-.8,.8,(2,N,3))
    JD,_=dD_dq(Q,P); err=np.abs(JD-fd(Q,P)).max()/np.abs(JD).max()
    print(N,p,phi,fs,"rel err",err)
W=rng.normal(size=(9,9)); P=Params([1.0],W=W,force_scale=.2); Q=rng.uniform(-.8,.8,(1,3,3))
print("W err",np.abs(dD_dq(Q,P)[0]-fd(Q,P)).max())
# paper numbers
P=Params(np.array([0.97]*20)); Q0=ensemble(20)
t=time.time(); l=lyap_max(Q0,P); print("lam 0.97", l.mean(), l.std(), time.time()-t,"s")
P=Params(np.array([5.2]*20)); print("lam 5.2", lyap_max(Q0,P).mean())
P=Params(np.array([0.97]*5)); t=time.time()
lam,fo,co=spectrum(ensemble(5),P,return_decomp=True); print(np.round(lam.mean(0),3), "sum",lam.sum(1).mean(),"fold",fo.mean(),"coup",co.mean(), time.time()-t,"s")
