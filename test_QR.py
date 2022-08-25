# %%
lfp = sim.lfp
v   = lfp[:,np.newaxis,:].shape



# %%
n,m,ntr = v.shape
ne      = ntr*(n-p)  # number of block equations of size m
nnp     = m*p+mcor   # number of parameter vectors of size m

# init K
K = np.zeros((ne,nnp+m))

if mcor == 1:
    K[:,0] = np.squeeze(np.ones((ne,1))) #TODO: find a better way to do this

# build K
for itr in range(1,ntr+1):
    for j in range(1,p+1):
        myarr = np.squeeze(v[(p-j+1)-1 : (n-j), :, (itr)-1]) # changes the indexing from python to matlab
        K[ ((n-p)*(itr-1) + 1)-1 : ((n-p)*itr), (mcor+m*(j-1)+1)-1 : (mcor+m*j)] = myarr.reshape((myarr.shape[0],1)) # TODO: check if this is correct
    
    myarr2 = np.squeeze(v[ (p+1)-1:n,:,itr-1 ])
    K[ ((n-p)*(itr-1) + 1)-1 : ((n-p)*itr), (nnp+1)-1 : nnp+m ] = myarr2.reshape((myarr2.shape[0],1))

q     = nnp + m  # number of columns of K

# times epsilon as floating point number precision
delta = (q**2 + q + 1) * np.finfo(np.float64).eps # Higham's choice for a Cholesky factorization
scale = np.sqrt(delta) * np.sqrt( np.sum(K**2,axis=0))
Q, R  = np.linalg.qr(np.vstack((K,np.diag(scale))),mode='complete')