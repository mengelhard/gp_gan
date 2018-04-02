import numpy as np

def pairwise_distance_np(x1,x2):
    return np.transpose((x1[np.newaxis,:]-x2[:,np.newaxis])**2)

def sample_normal(mean,cov,n_samples):
	l = np.linalg.cholesky(cov)
	samples = []
	dim = len(mean)
	for i in range(n_samples):
		samples.append(mean + l@np.random.normal(0,1,dim))
	return samples

def sgp_samples(x,y,m,t,sls,sfs,noise,n_samples=1):

	'''Sample from sparse GP
	x = inputs
	y = data
	m = pseudo-inputs
	t = gridpoints for the sample
	sls = length scale
	sfs = amplitude
	noise = noise
	n_samples = number of samples to draw'''

	kxm = sfs*np.exp(-.5*pairwise_distance_np(x,m)/sls)
	kmm = sfs*np.exp(-.5*pairwise_distance_np(m,m)/sls)
	kmx = kxm.T

	kmm_inv = np.linalg.inv(kmm)

	#gam_diag = [gp_params[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(len(x))]
	gam_diag = sfs - np.sum(np.matmul(kxm,kmm_inv)*kxm,axis=1)
	gam = np.diag(gam_diag)
	#gameyeinv = np.diag([1/(g+sigsq) for g in gam_diag])
	gameyeinv = np.diag(1/(gam_diag+noise))

	qm = kmm + kmx@gameyeinv@kxm
	qm_inv = np.linalg.inv(qm)

	ktm = sfs*np.exp(-.5*pairwise_distance_np(t,m)/sls)
	ktt = sfs*np.exp(-.5*pairwise_distance_np(t,t)/sls)
	kmt = ktm.T

	mean = ktm@qm_inv@kmx@gameyeinv@y
	cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + noise*np.identity(len(t))

	samples = sample_normal(mean,cov,n_samples)

	return samples