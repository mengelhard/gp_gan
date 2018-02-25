import numpy as np

def rand_poly(n_samples=200,order=5,max_coeff=1,domain=3,noise_var=1):

	coeffs = np.random.uniform(low=-max_coeff,high=max_coeff,size=order)

	x = np.random.uniform(low=-domain,high=domain,size=n_samples)
	noise = np.random.normal(0,noise_var,n_samples)
	y = noise + [sum([coeff*(sample**i) for i,coeff in enumerate(coeffs)]) for sample in x]

	samples = list(zip(x,y))

	return coeffs, x, y

def sample_gp(mean,cov,n_samples):

	l = np.linalg.cholesky(cov)

	samples = []

	dim = len(mean)

	for i in range(n_samples):

		samples.append(mean + np.matmul(l,np.random.normal(0,1,dim)))

	return samples