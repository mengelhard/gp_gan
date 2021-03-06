import numpy as np
import tensorflow as tf
from scipy.special import gammaln
import matplotlib.pyplot as plt
import itertools

#def pairwise_distance_np(x1,x2):
#	return np.transpose((x1[np.newaxis,:]-x2[:,np.newaxis])**2)

def pairwise_distance_np(x1,x2):
	r1 = np.sum(x1**2,axis=1)[:,np.newaxis]
	r2 = np.sum(x2**2,axis=1)[np.newaxis,:]
	r12 = x1@(x2.T)
	return r1+r2-2*r12

#def pairwise_distance(x1,x2):
#	# Does not currently support >1 dimensional coords (which would be axis 2 of inputs and axis 3 when adjusted)
#	return tf.matrix_transpose((tf.expand_dims(x1,1)-tf.expand_dims(x2,2))**2)

def pairwise_distance(x1,x2):
	r1 = tf.reduce_sum(x1**2,axis=2)[:,:,tf.newaxis]
	r2 = tf.reduce_sum(x2**2,axis=2)[:,tf.newaxis,:]
	r12 = tf.matmul(x1,tf.matrix_transpose(x2))
	return r1+r2-2*r12

def scaled_square_dist(x1,x2,sls):
	'''sls is a P by D matrix of squared length scales, 
	where P is the number of particles and D is the number of dimenions'''
	ls = tf.sqrt(sls)[:,tf.newaxis,:]
	return pairwise_distance(x1/ls,x2/ls)

def ssd(z,D):
    m = z[:,D+2:]
    m = tf.reshape(m,[tf.shape(m)[0],-1,D])
    centers = tf.reduce_mean(m,axis=1)
    sumsqdist = tf.reduce_sum((m-centers[:,tf.newaxis,:])**2,axis=[1,2])
    return sumsqdist

def scaled_square_dist_np(x1,x2,sls):
	'''sls is a P by D matrix of squared length scales, 
	where P is the number of particles and D is the number of dimenions'''
	ls = np.sqrt(sls)[np.newaxis,:]
	x1 = x1/ls
	x2 = x2/ls
	r1 = np.sum(x1**2,axis=1)[:,np.newaxis]
	r2 = np.sum(x2**2,axis=1)[np.newaxis,:]
	r12 = x1@(x2.T)
	return r1+r2-2*r12

def sample_normal(mean,cov,n_samples):
	l = np.linalg.cholesky(cov)
	samples = []
	dim = len(mean)
	for i in range(n_samples):
		samples.append(mean + l@np.random.normal(0,1,dim))
	return samples

def woodbury_inverse(ai,b,ci,d):
	m = np.linalg.inv(ci+d@ai@b)
	return ai-ai@b@m@d@ai

def loggamma(x,k,theta):
	return -k*np.log(theta)-gammaln(k)+(k-1)*tf.log(x)-x/theta

def lognormal(x,mean,var,logtp):
	return ((x-mean)**2)/(2*var)-.5*np.log(2*3.14*var)

def initialize_pips(boundaries,rpd,pprpd,ppr,jitter=1e-2):
	'''Initialize pseudo-inputs.
	boundaries is a list of intervals, one for each dimension
	rpd is the number of regions per dimension
	pprpd is the number of pseudo-inputs per rpd
	ppr is the number of particles per region'''
	rb = [intervals(b[0],b[1],rpd) for b in boundaries]
	regions = list(itertools.product(*rb))
	ip = [rpoints(r,pprpd) for r in regions]
	ip = np.tile(np.squeeze(np.array(ip).reshape(len(ip),-1,1)),(ppr,1))
	return ip + jitter*np.random.randn(*np.shape(ip))

def intervals(low,high,num):
	sz = (high-low)/num
	return [[low+sz*i,low+sz*(i+1)] for i in range(num)]

def rpoints(r,pprpd):
	coords = np.meshgrid(*[np.linspace(x[0],x[1],pprpd) for x in r])
	return np.hstack([x.reshape(-1,1) for x in coords])

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

	noise = np.abs(noise)

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
	#cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + noise*np.identity(len(t))
	cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + .000001*np.identity(len(t))

	samples = sample_normal(mean,cov,n_samples)

	return samples

def sgp_pred(x,y,t,z):

	D = np.shape(x)[1]
	
	sls = z[:D]
	sfs = z[D]
	noise = z[D+1]

	m = z[D+2:]
	m = np.reshape(m,[-1,D]) # unflatten pseudo-inputs

	#kxm = sfs*np.exp(-.5*pairwise_distance_np(x,m)/sls)
	#kmm = sfs*np.exp(-.5*pairwise_distance_np(m,m)/sls)
	kxm = sfs*np.exp(-.5*scaled_square_dist_np(x,m,sls))
	kmm = sfs*np.exp(-.5*scaled_square_dist_np(m,m,sls))
	kmx = kxm.T

	jitter = 1e-8*np.eye(len(m))

	kmm_inv = np.linalg.inv(kmm+jitter)

	#gam_diag = [gp_params[1]-kmx[:,i].T@kmm_inv@kmx[:,i] for i in range(len(x))]
	gam_diag = sfs - np.sum(np.matmul(kxm,kmm_inv)*kxm,axis=1)
	gam = np.diag(gam_diag)
	#gameyeinv = np.diag([1/(g+sigsq) for g in gam_diag])
	gameyeinv = np.diag(1/(gam_diag+noise))

	qm = kmm + kmx@gameyeinv@kxm
	qm_inv = np.linalg.inv(qm+jitter)

	#ktm = sfs*np.exp(-.5*pairwise_distance_np(t,m)/sls)
	#ktt = sfs*np.exp(-.5*pairwise_distance_np(t,t)/sls)
	ktm = sfs*np.exp(-.5*scaled_square_dist_np(t,m,sls))
	ktt = sfs*np.exp(-.5*scaled_square_dist_np(t,t,sls))
	kmt = ktm.T

	mean = ktm@qm_inv@kmx@gameyeinv@y
	cov = ktt - ktm@(kmm_inv - qm_inv)@kmt + noise*np.identity(len(t))

	return mean, cov

def sgp_prediction_old(x,y,m,t,sls,sfs,noise):

	noise = np.abs(noise)

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

	return mean, cov

def multi_prediction(x,y,t,particles):
	'''return average mean and covariance for all particles'''
	mean = []
	cov = []

	for p in particles:
		p_mean,p_cov = sgp_prediction(x,y,p[3:],t,*p[:3])
		mean.append(p_mean)
		cov.append(p_cov)

	mean = np.mean(np.array(mean),axis=0)
	assert len(mean) == len(t)

	cov = np.mean(np.array(cov),axis=0)
	assert np.shape(cov) == (len(t),len(t))

	return mean, cov

def nlog_gammaprior(z,alpha,beta):
	beta = np.array(beta)[tf.newaxis,:]
	logg = alpha*np.log(beta)-gammaln(alpha)+(alpha-1)*tf.log(z)-beta*z
	return -tf.reduce_sum(logg,axis=1)

def nlog_fitc(x,y,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles
	#x = tf.expand_dims(x,0)
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(P,1,1))
	#x = x[tf.newaxis,:,:]
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))
	
	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	
	sls = tf.transpose(z[:D])
	sfs = z[D]
	noise = z[D+1]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[D+2:])
	m = tf.reshape(m,[tf.shape(m)[0],-1,D]) # unflatten pseudo-inputs
	m = m + 1e-6*tf.random_normal(tf.shape(m),dtype=tf.float64)

	M = tf.shape(m)[1]

	jitter = 1e-8*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	#xm = pairwise_distance(x,m)
	#mm = pairwise_distance(m,m)

	xm = scaled_square_dist(x,m,sls)
	mm = scaled_square_dist(m,m,sls)

	#kxm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	#kmm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	
	#kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	#kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm)
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm)
	kmx = tf.matrix_transpose(kxm)
	
	kmmi = tf.linalg.inv(kmm+jitter)
	#kmmi = tf.matrix_inverse(kmm+jitter)

	qmm = tf.matmul(tf.matmul(kxm,kmmi),kmx)
	
	gd = (sfs+noise)[:,tf.newaxis] - tf.matrix_diag_part(qmm)
	gid = 1/gd
	g = tf.matrix_diag(gd)
	gi = tf.matrix_diag(gid)
	
	cov = qmm+g
	#kmx_gi_kxm = tf.matmul(tf.matmul(kmx,gi),kxm)
	kmx_gi_kxm = tf.matmul(rmult(kmx,gid),kxm)
	#covi = tf.matrix_inverse(cov)
	#covi = gi - tf.matmul(gi,tf.matmul(kxm,tf.matmul(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gi)))
	covi = gi - lmult(gid,tf.matmul(kxm,rmult(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gid)))
	#covd = tf.linalg.logdet(cov)
	covd = tf.linalg.logdet(kmmi+kmx_gi_kxm+jitter)+tf.linalg.logdet(kmm+jitter)+tf.reduce_sum(tf.log(gd),axis=1)
	#covd = tf.log(tf.matrix_determinant(kmmi+kmx_gi_kxm))+tf.log(tf.matrix_determinant(kmm))+tf.reduce_sum(tf.log(gd),axis=1)

	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.squeeze(tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	t3 = .5*covd
	
	return t1+t2+t3

def nlog_vfe(x,y,z):

	z = tf.transpose(z)

	P = tf.shape(z)[1] # number of particles
	#x = tf.expand_dims(x,0)
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(P,1,1))
	#x = x[tf.newaxis,:,:]
	x = tf.tile(tf.expand_dims(x,0),(P,1,1))

	D = tf.shape(x)[2]
	X = tf.shape(x)[1]
	
	sls = tf.transpose(z[:D])
	sfs = z[D]
	noise = z[D+1]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[D+2:])
	m = tf.reshape(m,[tf.shape(m)[0],-1,D]) # unflatten pseudo-inputs
	m = m + 1e-6*tf.random_normal(tf.shape(m),dtype=tf.float64)

	M = tf.shape(m)[1]

	jitter = 1e-8*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	#xm = pairwise_distance(x,m)
	#mm = pairwise_distance(m,m)

	xm = scaled_square_dist(x,m,sls)
	mm = scaled_square_dist(m,m,sls)

	#kxm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	#kmm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	
	#kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	#kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm)
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm)
	
	kmx = tf.matrix_transpose(kxm)
	#kmmi = tf.linalg.inv(kmm+jitter)
	kmmi = tf.matrix_inverse(kmm+jitter)

	qmm = tf.matmul(tf.matmul(kxm,kmmi),kmx)
	
	#g = tf.matrix_diag(gd)
	g = noise[:,tf.newaxis,tf.newaxis]*tf.eye(X,dtype=tf.float64)[tf.newaxis,:,:]
	#gi = tf.matrix_diag(1/gd)
	gi = (1/noise)[:,tf.newaxis,tf.newaxis]*tf.eye(X,dtype=tf.float64)[tf.newaxis,:,:]
	gd = noise[:,tf.newaxis]
	gid = 1/gd

	tr = tf.reduce_sum(sfs[:,tf.newaxis] - tf.matrix_diag_part(qmm),axis=1)
	
	cov = qmm+g
	#kmx_gi_kxm = tf.matmul(tf.matmul(kmx,gi),kxm)
	kmx_gi_kxm = tf.matmul(rmult(kmx,(1/gd)),kxm)
	#covi = tf.matrix_inverse(cov)
	#covi = gi - tf.matmul(gi,tf.matmul(kxm,tf.matmul(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gi)))
	covi = gi - lmult(gid,tf.matmul(kxm,rmult(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gid)))
	#covd = tf.linalg.logdet(cov)
	covd = tf.linalg.logdet(kmmi+kmx_gi_kxm+jitter)+tf.linalg.logdet(kmm+jitter)+tf.log(noise)*tf.cast(X,dtype=tf.float64)#tf.reduce_sum(tf.log(gd),axis=1)
	#covd = tf.log(tf.matrix_determinant(kmmi+kmx_gi_kxm))+tf.log(tf.matrix_determinant(kmm))+tf.log(gd)*tf.cast(X,dtype=tf.float64)#tf.reduce_sum(tf.log(gd),axis=1)

	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.squeeze(tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	t3 = .5*covd
	t4 = .5*tf.div(tr,noise)
	
	return t1+t2+t3

def nlog_vfe_old(x,y,z):

	# need to make dimensionality changes like fitc (above)

	z = tf.transpose(z)
	x = tf.expand_dims(x,0)
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(tf.shape(z)[1],1,1))
	
	sls = z[0]
	sfs = z[1]
	noise = z[2]

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	m = tf.transpose(z[3:])
	m = m + 1e-6*tf.random_normal(tf.shape(m),dtype=tf.float64)

	X = tf.shape(x)[-1]
	M = tf.shape(m)[-1]

	jitter = 1e-8*tf.eye(M,dtype=tf.float64)[tf.newaxis,:,:]

	xm = pairwise_distance(x,m)
	mm = pairwise_distance(m,m)

	#kxm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	#kmm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	
	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	
	kmx = tf.matrix_transpose(kxm)
	kmmi = tf.linalg.inv(kmm+jitter)

	qmm = tf.matmul(tf.matmul(kxm,kmmi),kmx)
	
	#g = tf.matrix_diag(gd)
	g = noise[:,tf.newaxis,tf.newaxis]*tf.eye(X,dtype=tf.float64)[tf.newaxis,:,:]
	#gi = tf.matrix_diag(1/gd)
	gi = (1/noise)[:,tf.newaxis,tf.newaxis]*tf.eye(X,dtype=tf.float64)[tf.newaxis,:,:]
	gd = noise[:,tf.newaxis]
	gid = 1/gd

	tr = tf.reduce_sum(sfs[:,tf.newaxis] - tf.matrix_diag_part(qmm),axis=1)
	
	cov = qmm+g
	#kmx_gi_kxm = tf.matmul(tf.matmul(kmx,gi),kxm)
	kmx_gi_kxm = tf.matmul(rmult(kmx,(1/gd)),kxm)
	#covi = tf.matrix_inverse(cov)
	#covi = gi - tf.matmul(gi,tf.matmul(kxm,tf.matmul(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gi)))
	covi = gi - lmult(gid,tf.matmul(kxm,rmult(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gid)))
	#covd = tf.linalg.logdet(cov)
	covd = tf.linalg.logdet(kmmi+kmx_gi_kxm)+tf.linalg.logdet(kmm)+tf.log(gd)*tf.cast(X,dtype=tf.float64)#tf.reduce_sum(tf.log(gd),axis=1)

	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.squeeze(tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	t3 = .5*covd
	t4 = .5*tf.div(tr,noise)
	
	return t1+t2+t3

def lmult(diag,mat,naxes=3):
	return tf.expand_dims(diag,axis=naxes-1)*mat

def rmult(mat,diag,naxes=3):
	return mat*tf.expand_dims(diag,axis=naxes-2)

def sgp(x,y,z):
	
	# Note: not working well with csum=False. pseudo-inputs go crazy. something wrong with barrier function?

	x_min = np.amin(x)
	pip_noise = 1e-6*np.std(x)
		
	z = tf.transpose(z)
	x = tf.expand_dims(x,0)
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(tf.shape(z)[1],1,1))
	
	sls = z[0]#[:,tf.newaxis,tf.newaxis]
	sfs = z[1]#[:,tf.newaxis,tf.newaxis]
	#noise = z[2]#[:,tf.newaxis,tf.newaxis]
	noise = z[2]#[:,tf.newaxis,tf.newaxis]
	#pip_noise = np.abs(z[3])
	#pip_noise = 1e-10

	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)

	params_prior = loggamma(z[0],2,1)+loggamma(z[1],2,1)+loggamma(z[2],2,.1)
	#pips_prior = tf.reduce_sum(lognormal(z[3:],np.mean(x),np.var(x),logtp),axis=0)

	m = tf.transpose(z[3:])
	m = m + tf.random_normal(tf.shape(m),dtype=tf.float64)*pip_noise#[:,tf.newaxis] # add noise to pips to avoid colocation
	
	xm = pairwise_distance(x,m)#[tf.newaxis,:,:]
	mm = pairwise_distance(m,m)#[tf.newaxis,:,:]
	
	kxm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	kmm = (noise*sfs)[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
	
	kmx = tf.matrix_transpose(kxm)
	kmmi = tf.linalg.inv(kmm)
	
	X = tf.shape(x)[-1]
	M = tf.shape(m)[-1]
	
	#ge = sfs[:,tf.newaxis] + noise[:,tf.newaxis] - tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)
	ge = (noise*(sfs+1))[:,tf.newaxis] - tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)
	gei = 1./ge
	
	kmxgeikxm = tf.matmul(kmx,gei[:,:,tf.newaxis]*kxm)
	
	# remove these for now
	# alternative from stack overflow
	
	#one = tf.matrix_inverse(tf.eye(M,dtype=tf.float64)+tf.matmul(tf.matmul(tf.matmul(kmx,gei_mat),kxm),kmmi))
	#two = tf.matmul(kxm,tf.matmul(kmmi,tf.matmul(one,tf.matmul(kmx,gei_mat))))
	#one = tf.matrix_inverse(kmm+tf.matmul(kmx,tf.matmul(gei_mat,kxm)))
	#two = tf.matmul(tf.matmul(tf.matmul(kxm,one),kmx),gei_mat)
	#covi = tf.matmul(gei_mat,tf.eye(X,dtype=tf.float64)-two)
	
	#covd = tf.linalg.logdet(kmm+kmxgeikxm)+tf.linalg.logdet(kmmi)+tf.reduce_sum(tf.log(ge),axis=1)
	#covi = tf.matrix_diag(gei)-gei[:,:,tf.newaxis]*tf.matmul(kxm,(tf.matmul(tf.matrix_inverse(kmm+kmxgeikxm),kmx))*gei[:,tf.newaxis,:]) # inverse
	#covi = gei_mat-tf.matmul(tf.matmul(gei_mat,tf.matmul(tf.matmul(kxm,tf.matrix_inverse(kmm+kmxgeikxm)),kmx)),gei_mat) # inverse
	
	# alternative, more computationally intensive
	
	cov = tf.matmul(tf.matmul(kxm,kmmi),kmx)+tf.matrix_diag(ge)
	covi = tf.matrix_inverse(cov)
	covd = tf.linalg.logdet(cov)

	top = tf.squeeze(-.5*tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	bot = .5*tf.cast(X,tf.float64)*logtp+.5*covd
	
	return bot-top-params_prior#-pips_prior

def sgp_old(x,y,z,pips=True,csum=True):
	
	# Note: not working well with csum=False. pseudo-inputs go crazy. something wrong with barrier function?
		
	x_min = np.amin(x)
		
	z = tf.transpose(z)
	x = tf.expand_dims(x,0)
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(tf.shape(z)[1],1,1))
	
	sls = z[0]#[:,tf.newaxis,tf.newaxis]
	sfs = z[1]#[:,tf.newaxis,tf.newaxis]
	noise = z[2]#[:,tf.newaxis,tf.newaxis]
	#noise = .000001

	if pips:
		if csum:
			m = x_min+tf.cumsum(tf.transpose(z[3:]),axis=1) # start with minimum val of x
		else:
			m = tf.transpose(z[3:])
	else:
		m=np.linspace(2,9,10)[tf.newaxis,:]
		#m = np.array([2.,3.,4.,5.,6.,7.,8.,9.,10.])[tf.newaxis,:]
	
	xm = pairwise_distance(x,m)#[tf.newaxis,:,:]
	mm = pairwise_distance(m,m)#[tf.newaxis,:,:]
	
	if csum:
		#barrier = tf.reduce_sum(tf.log(z),axis=0)
		barrier = -tf.log(tf.reduce_sum(1/z,axis=0))
	else:
		barrier = -tf.log(tf.reduce_sum(1/mm,axis=[1,2]))-tf.log(tf.reduce_sum(1/z[:2],axis=0))
	
	kxm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*xm/sls[:,tf.newaxis,tf.newaxis])
	kmm = sfs[:,tf.newaxis,tf.newaxis]*tf.exp(-.5*mm/sls[:,tf.newaxis,tf.newaxis])
		
	kmx = tf.matrix_transpose(kxm)
	kmmi = tf.linalg.inv(kmm)
	
	X = tf.shape(x)[-1]
	M = tf.shape(m)[-1]
	
	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)
	
	#ge = sfs[:,tf.newaxis] + noise - tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2) # gamma+noise*I diagonal
	ge = sfs[:,tf.newaxis] + noise[:,tf.newaxis] - tf.reduce_sum(tf.matmul(kxm,kmmi)*kxm,axis=2)
	gei = 1./ge
	
	#gei_mat = tf.matrix_inverse(tf.matrix_diag(ge))
	
	kmxgeikxm = tf.matmul(kmx,gei[:,:,tf.newaxis]*kxm)
	#kmxgeikxm = tf.matmul(kmx,tf.matmul(gei_mat,kxm))
	
	# remove these for now
	# alternative from stack overflow
	
	#one = tf.matrix_inverse(tf.eye(M,dtype=tf.float64)+tf.matmul(tf.matmul(tf.matmul(kmx,gei_mat),kxm),kmmi))
	#two = tf.matmul(kxm,tf.matmul(kmmi,tf.matmul(one,tf.matmul(kmx,gei_mat))))
	#one = tf.matrix_inverse(kmm+tf.matmul(kmx,tf.matmul(gei_mat,kxm)))
	#two = tf.matmul(tf.matmul(tf.matmul(kxm,one),kmx),gei_mat)
	#covi = tf.matmul(gei_mat,tf.eye(X,dtype=tf.float64)-two)
	
	#covd = tf.linalg.logdet(kmm+kmxgeikxm)+tf.linalg.logdet(kmmi)+tf.reduce_sum(tf.log(ge),axis=1)
	#covi = tf.matrix_diag(gei)-gei[:,:,tf.newaxis]*tf.matmul(kxm,(tf.matmul(tf.matrix_inverse(kmm+kmxgeikxm),kmx))*gei[:,tf.newaxis,:]) # inverse
	#covi = gei_mat-tf.matmul(tf.matmul(gei_mat,tf.matmul(tf.matmul(kxm,tf.matrix_inverse(kmm+kmxgeikxm)),kmx)),gei_mat) # inverse
	
	# alternative, more computationally intensive
	
	cov = tf.matmul(tf.matmul(kxm,kmmi),kmx)+tf.matrix_diag(ge)
	covi = tf.matrix_inverse(cov)
	covd = tf.linalg.logdet(cov)

	top = tf.squeeze(-.5*tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	bot = .5*tf.cast(X,tf.float64)*logtp+.5*covd
	
	# add barrier function
	
	return bot-top-barrier#, ge

def gp_np(x,y,z):
	
	z = z.T
	ym = np.tile(np.expand_dims(np.expand_dims(y,0),2),(np.shape(z)[1],1,1))
	
	sls = np.abs(z[0])[:,np.newaxis,np.newaxis]
	sfs = np.abs(z[1])[:,np.newaxis,np.newaxis]
	#noise = (np.abs(z[2])+.1)[:,np.newaxis,np.newaxis]
	noise = .000001
		
	xx = pairwise_distance_np(x,x)[np.newaxis,:,:]
	kxx = sfs*np.exp(-.5*xx/sls)
	
	X = np.shape(xx)[-1]
	
	logtp = np.log(2.*np.pi)
	
	cov = kxx+noise*np.eye(X)[np.newaxis,:,:]
	top = np.squeeze(-.5*np.matmul(np.matmul(ym.transpose((0,2,1)),np.linalg.inv(cov)),ym))
	bot = .5*X*logtp+.5*np.linalg.slogdet(cov)[1]
	
	return bot-top

def gp(x,y,z):
	
	z = tf.transpose(z)
	x = tf.expand_dims(x,0)
	ym = tf.tile(tf.expand_dims(tf.expand_dims(y,0),2),(tf.shape(z)[1],1,1))
	
	sls = z[0][:,tf.newaxis,tf.newaxis]
	#sfs = z[1][:,tf.newaxis,tf.newaxis]
	sfs = 1
	#noise = (tf.abs(z[2])+.1)[:,tf.newaxis,tf.newaxis]
	#noise = .000001
	noise = z[1][:,tf.newaxis,tf.newaxis]
		
	xx = pairwise_distance(x,x)#[tf.newaxis,:,:]
	kxx = sfs*tf.exp(-.5*xx/sls)
	
	X = tf.shape(x)[-1]
	
	logtp = tf.constant(np.log(2.*np.pi),dtype=tf.float64)
	
	cov = kxx+noise*tf.eye(X,dtype=tf.float64)#[tf.newaxis,:,:]

	top = tf.squeeze(-.5*tf.matmul(tf.matmul(tf.matrix_transpose(ym),tf.matrix_inverse(cov)),ym))
	bot = .5*tf.cast(X,tf.float64)*logtp+.5*tf.linalg.logdet(cov)
	
	return bot-top

def plot_pdf(U_z,npoints,*args,uselog=False):
	mesh_z1, mesh_z2, points = gridpoints(npoints,*args)
	z_pp  = tf.placeholder(tf.float64, [None, len(args)])
	if uselog:
		#out = U_z(z_pp)
		#prob = -out[0]
		#ge = out[1]
		prob = -U_z(z_pp)
	else:
		#out = U_z(z_pp)
		#prob = tf.exp(-out[0])
		#ge = out[1]
		prob = tf.exp(-U_z(z_pp))
	with tf.Session() as s:
		#phat_z, ge_val = s.run([prob, ge], feed_dict={z_pp: points} ) 
		phat_z = s.run(prob, feed_dict={z_pp: points} ) 
	print(phat_z)
	phat_z=phat_z.reshape([npoints,npoints])
	#phat_z=phat_z/np.abs(phat_z).max()
	plt.pcolormesh(mesh_z1, mesh_z2, phat_z)
	if uselog:
		z_min, z_max = np.nanmin(phat_z)-2*(np.nanmax(phat_z)-np.nanmin(phat_z)), np.nanmax(phat_z)
	else:
		z_min, z_max = -np.nanmax(phat_z), np.nanmax(phat_z)
	#print(phat_z,z_min,z_max)
	plt.pcolor(mesh_z1, mesh_z2, phat_z, cmap='RdBu', vmin=z_min, vmax=z_max)
	plt.xlim(args[0]); plt.ylim(args[1]); plt.title('Target distribution: $u(z)$')
	
	#return ge_val

def gridpoints(npoints,*args):
	sides = [np.linspace(arg[0],arg[1],npoints) for arg in args]
	coords = np.meshgrid(*sides)
	return coords[0], coords[1], np.hstack([coord.reshape(-1,1) for coord in coords])