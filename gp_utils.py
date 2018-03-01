import numpy as np
import itertools

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