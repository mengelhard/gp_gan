import numpy as np
import itertools
from scipy.stats import multivariate_normal

def pairwise_distance(x1, x2):
	# Broadcasting tricks to get every pairwise distance.
	return ((x1[np.newaxis, :, :] - x2[:, np.newaxis, :]) ** 2).sum(2).T

def sample_poly(coeffs=None,max_coeff=5,n_samples=200,order=5,domain=[-2,2],noise_var=10,sdist='uniform'):

	if coeffs is None:
		coeffs = np.random.choice(np.arange(-max_coeff,max_coeff),size=order)

	if sdist=='norm':
		x = np.random.normal(0,domain[1],size=n_samples)
	else:
		x = np.random.uniform(low=domain[0],high=domain[1],size=n_samples)
	noise = np.random.normal(0,np.sqrt(noise_var),n_samples)
	y = noise + [sum([coeff*(sample**i) for i,coeff in enumerate(coeffs)]) for sample in x]

	samples = list(zip(x,y))

	return coeffs, x, y

def sample_normal(mean,cov,n_samples):

	l = np.linalg.cholesky(cov)

	samples = []

	dim = len(mean)

	for i in range(n_samples):

		samples.append(mean + l@np.random.normal(0,1,dim))

	return samples

#def sample_gp(x,cov_kernel,n_samples=1,data=None):

	#kxx = cov_kernel(x,x)

def sqe_kernel(x1,x2,siglsq,sigfsq):

	out = np.zeros((len(x1),len(x2)))

	for i in range(len(x1)):
		for j in range(len(x2)):
			d = x2[j]-x1[i]
			out[i,j] = sigfsq*np.exp(-.5*np.dot(d,d)/siglsq)

	return out

def sparse_gp(xn,yn,xm,x,sigsq,siglsq,sigfsq):
	'''Sparse GP regression
	data xn, yn
	pseudo-inputs xm
	evaluate at x
	noise variance sigsq
	kernel parameters siglsq, sigfsq'''

	N = len(xn)
	M = len(xm)

	knm = sqe_kernel(xn,xm,siglsq,sigfsq)
	kmm = sqe_kernel(xm,xm,siglsq,sigfsq)
	#kmn = sqe_kernel(xm,xn,siglsq,sigfsq)
	kmn = knm.T

	#print(knm,kmm,kmn)

	assert np.shape(knm) == (N,M)
	assert np.shape(kmm) == (M,M)
	assert np.shape(kmn) == (M,N)

	kmminv = np.linalg.inv(kmm)

	#print(kmminv)

	gam_diag = [sigfsq-kmn[:,i].T@kmm_inv@kmn[:,i] for i in range(N)]
	gam = np.diag(gam_diag)
	gameyeinv = np.diag([1/(g+sigsq) for g in gam_diag])

	#gam = np.zeros((N,N))
	#gameyeinv = np.zeros((N,N))

	#for i in range(N):
		#kn = sqe_kernel(xm,[xn[i]],siglsq,sigfsq)
		#g = sqe_kernel([xn[i]],[xn[i]],siglsq,sigfsq) - kn.T@kmminv@kn
		#gam[i,i] = g
		#gameyeinv[i,i] = 1/(g+sigsq)

	#print(gam)
	#print(gameyeinv)

	qm = kmm + kmn@gameyeinv@knm
	qminv = np.linalg.inv(qm)

	#print(qm,qminv)

	assert np.shape(qm) == (M,M)

	meanmat = qminv@kmn@gameyeinv@yn
	covmat = kmminv-qminv

	y = np.zeros(len(x))

	for i, sample in enumerate(x):
		ks = sqe_kernel(xm,[sample],siglsq,sigfsq)
		ss = sqe_kernel([sample],[sample],siglsq,sigfsq) - ks.T@covmat@ks + sigsq
		ms = ks.T@meanmat
		#print(sample,ms,ss)
		y[i] = np.random.normal(ms,np.sqrt(ss))

	return y

def sparse_gp_qm(xn,yn,xm,x,sigsq,siglsq,sigfsq):
	'''Sparse GP regression
	data xn, yn
	pseudo-inputs xm
	evaluate at x
	noise variance sigsq
	kernel parameters siglsq, sigfsq'''

	N = len(xn)
	M = len(xm)

	knm = sqe_kernel(xn,xm,siglsq,sigfsq)
	kmm = sqe_kernel(xm,xm,siglsq,sigfsq)
	kmn = sqe_kernel(xm,xn,siglsq,sigfsq)

	#print(knm,kmm,kmn)

	assert np.shape(knm) == (N,M)
	assert np.shape(kmm) == (M,M)
	assert np.shape(kmn) == (M,N)

	kmminv = np.linalg.inv(kmm)

	#print(kmminv)

	gam = np.zeros((N,N))
	gameyeinv = np.zeros((N,N))

	for i in range(N):
		kn = sqe_kernel(xm,[xn[i]],siglsq,sigfsq)
		g = sqe_kernel([xn[i]],[xn[i]],siglsq,sigfsq) - kn.T@kmminv@kn
		gam[i,i] = g
		gameyeinv[i,i] = 1/(g+sigsq)

	#print(gam)
	#print(gameyeinv)

	qm = kmm + kmn@gameyeinv@knm
	qminv = np.linalg.inv(qm)

	#print(qm,qminv)

	assert np.shape(qm) == (M,M)

	meanmat = qminv@kmn@gameyeinv@yn
	covmat = kmminv-qminv

	y = np.zeros(len(x))

	for i, sample in enumerate(x):
		ks = sqe_kernel(xm,[sample],siglsq,sigfsq)
		ss = sqe_kernel([sample],[sample],siglsq,sigfsq) - ks.T@covmat@ks + sigsq
		ms = ks.T@meanmat
		#print(sample,ms,ss)
		y[i] = np.random.normal(ms,np.sqrt(ss))

	return y

def slice_sampler(f,w,x,n):
	'''Sample from f using slice sampling:
	w: width parameter
	x: initial point
	n: number of samples

	one dimensional for now'''

	w = np.array(w)
	x = np.array(x)

	samples = []

	for _ in range(n):

		x_old = x

		y = np.random.uniform(high=f(x))
		x_max = x+w
		x_min = x-w

		while f(x_max)>y:
			x_max += w

		while f(x_min)>y:
			x_min -= w

		x = np.random.uniform(low=x_min,high=x_max)

		while f(x)<y:

			pos_dims = x>x_old
			neg_dims = x<x_old

			x_max[pos_dims] = x[pos_dims]
			x_min[neg_dims] = x[neg_dims]

			x = np.random.uniform(low=x_min,high=x_max)

		samples.append(x)

	return samples

def sgp_slice_sampler(x,y,hps,widths,n_samples=1,step_out=True):
	"""efficiently slice sample from sparse GP likelihood

	 Inputs:
				 x  inputs
				 y  observations
			   hps  hyperparameters, where hps = [sls,sfs,noise,m]:
			   sls  length scale
			   sfs  scale
			 noise  observation noise
				 m  pseudo-inputs
		 n_samples  number of samples to draw
			widths  slice sampling step size for each element of hps
		  step_out bool set to True (default) if widths sometimes far too small

	 Outputs:
		   samples  of form [sls, sfs, noise, m] and length n_samples
	"""

	# startup stuff
	hps = np.array(hps)
	widths = np.array(widths)

	assert hps.shape == widths.shape

	H = hps.size
	X = x.size

	perm = np.array(range(H))[widths>0] # 3 non pip params

	# Calculate quantities based on initial values

	xm = pairwise_dist(x,hps[3:])
	mm = pairwise_dist(hps[3:],hps[3:])

	kxm = hps[1]*np.exp(-.5*xm/hps[0])
	kmm = hps[1]*np.exp(-.5*mm/hps[0])

	kmm_inv = np.linalg.inv(kmm)

	gam = np.diag([hps[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(X)])

	cov = kxm@kmm_inv@kxm.T + gam + hps[2]*np.eye(X)

	log_Px = gaussian_logpdf(y,cov)

	hps_l = hps.copy()
	hps_r = hps.copy()
	hps_prime = hps.copy()

	samples = []

	# start sampling!
	for i in range(n_samples):

		np.random.shuffle(perm)

		for dd in perm:

			hps_old = hps

			log_uprime = log_Px + np.log(np.random.rand())
			# Create a horizontal interval (x_l, x_r) enclosing xx
			rr = np.random.rand()
			hps_l[dd] = hps[dd] - rr*widths[dd]
			hps_r[dd] = hps[dd] + (1-rr)*widths[dd]

			if step_out:
				# Typo in early book editions: said compare to u, should be u'
				while log_Px > log_uprime:
					cov = update_cov(xm,mm,kxm,kmm,kmm_inv,gam,cov,hps_l,hps_old,dd)
					log_Px = gaussian_logpdf(y,cov)
					hps_l[dd] = hps_l[dd] - widths[dd]
				while log_Px > log_uprime:
					cov = update_cov(xm,mm,kxm,kmm,kmm_inv,gam,cov,hps_r,hps_old,dd)
					log_Px = gaussian_logpdf(y,cov)
					hps_r[dd] = hps_r[dd] + widths[dd]

			# Inner loop:
			# Propose xprimes and shrink interval until good one found
			while True:
				hps_prime[dd] = np.random.rand()*(hps_r[dd] - hps_l[dd]) + hps_l[dd]
				xm,mm,kxm,kmm,kmm_inv,gam,cov = update_cov(xm,mm,kxm,kmm,
					kmm_inv,gam,cov,hps_prime,dd,hps_old,return_all=True)
				log_Px = gaussian_logpdf(y,cov)
				if log_Px > log_uprime:
					break # this is the only way to leave the while loop
				else:
					# Shrink in
					if hps_prime[dd] > hps[dd]:
						hps_r[dd] = hps_prime[dd]
					elif hps_prime[dd] < hps[dd]:
						hps_l[dd] = hps_prime[dd]
					else:
						raise Exception('BUG DETECTED: Shrunk to current '
							+ 'position and still not acceptable.')
			hps[dd] = hps_prime[dd]
			hps_l[dd] = hps_prime[dd]
			hps_r[dd] = hps_prime[dd]

		samples.append(xx)

	return np.vstack(samples)

def gaussian_logpdf(y,cov,eps=0.000001):
	try:
		return multivariate_normal.logpdf(y,cov=cov+eps*np.eye(len(y)))
	except:
		return float('-inf')

def update_cov(xm,mm,kxm,kmm,kmm_inv,gam,cov,hps,dd,hps_old,return_all=False):

	# would help to have the old hps!!

	# intelligently update all this stuff based on the known changes to hps
	if dd==0:
		# update length param
		kxm = hps[1]*np.exp(-.5*xm/hps[0])
		kmm = hps[1]*np.exp(-.5*mm/hps[0])

		kmm_inv = np.linalg.inv(kmm)

		gam = np.diag([hps[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(X)])

		cov = kxm@kmm_inv@kxm.T + gam + hps[2]*np.eye(X)

	elif dd==1:
		# update for scale
		# shortcut to kmm_inv update
		kxm = kxm*(hps[1]/hps_old[1])
		kmm = kmm*(hps[1]/hps_old[1])

		kmm_inv = kmm_inv*(hps_old[1]/hps[1])

		#gam = np.diag([hps[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(X)])
		gam = gam*(hps[1]/hps_old[1])

		cov = kxm@kmm_inv@kxm.T + gam + hps[2]*np.eye(X)

	elif dd==2:
		# update for noise
		# very little to do here
		cov = cov + (hps[2]-hps_old[2])*np.eye(X)
	else:
		# update for appropriate pip
		xm_new = pairwise_dist(x,hps[dd])
		xm[:,dd-3] = xm_new
		mm_new = pairwise_dist(hps[3:],hps[dd])
		mm[:,dd-3] = mm_new
		mm[dd-3,:] = mm_new

		kxm[:,dd-3] = hps[1]*np.exp(-.5*xm_new/hps[0])
		kmm[:,dd-3] = hps[1]*np.exp(-.5*mm_new/hps[0])
		kmm[dd-3,:] = hps[1]*np.exp(-.5*mm_new/hps[0])

		kmm_inv = np.linalg.inv(kmm)

		gam = np.diag([hps[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(X)])

		cov = kxm@kmm_inv@kxm.T + gam + hps[2]*np.eye(X)

	if return_all:
		return xm,mm,kxm,kmm,kmm_inv,gam,cov
	else:
		return cov