import numpy as np
import tensorflow as tf
from scipy.special import gammaln
import matplotlib.pyplot as plt

def pairwise_distance_np(x1,x2):
	return np.transpose((x1[np.newaxis,:]-x2[:,np.newaxis])**2)

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

def loggamma(x,alpha,beta):
	return alpha*np.log(beta)-gammaln(alpha)+(alpha-1)*tf.log(x)-beta*x

def lognormal(x,mean,var,logtp):
	return ((x-mean)**2)/(2*var)-.5*np.log(2*3.14*var)

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

def sgp_prediction(x,y,m,t,sls,sfs,noise):

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

def nlog_gammaprior(z,alpha,beta):
	beta = np.array(beta)[tf.newaxis,:]
	logg = alpha*np.log(beta)-gammaln(alpha)+(alpha-1)*tf.log(z)-beta*z
	return -tf.reduce_sum(logg,axis=1)

def nlog_fitc(x,y,z):

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
	
	gd = (sfs+noise)[:,tf.newaxis] - tf.matrix_diag_part(qmm)
	g = tf.matrix_diag(gd)
	gi = tf.matrix_diag(1/gd)
	
	cov = qmm+g
	kmx_gi_kxm = tf.matmul(tf.matmul(kmx,gi),kxm)
	#covi = tf.matrix_inverse(cov)
	covi = gi - tf.matmul(gi,tf.matmul(kxm,tf.matmul(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gi)))
	#covd = tf.linalg.logdet(cov)
	covd = tf.linalg.logdet(kmmi+kmx_gi_kxm)+tf.linalg.logdet(kmm)+tf.reduce_sum(tf.log(gd),axis=1)

	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.squeeze(tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	t3 = .5*covd
	
	return t1+t2+t3

def nlog_vfe(x,y,z):

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
	
	gd = noise[:,tf.newaxis]
	g = tf.matrix_diag(gd)
	gi = tf.matrix_diag(1/gd)

	tr = tf.reduce_sum(sfs[:,tf.newaxis] - tf.matrix_diag_part(qmm),axis=1)
	
	cov = qmm+g
	kmx_gi_kxm = tf.matmul(tf.matmul(kmx,gi),kxm)
	#covi = tf.matrix_inverse(cov)
	covi = gi - tf.matmul(gi,tf.matmul(kxm,tf.matmul(tf.matmul(tf.matrix_inverse(kmmi+kmx_gi_kxm+jitter),kmx),gi)))
	#covd = tf.linalg.logdet(cov)
	covd = tf.linalg.logdet(kmmi+kmx_gi_kxm)+tf.linalg.logdet(kmm)+tf.reduce_sum(tf.log(gd),axis=1)

	t1 = .5*tf.cast(X,tf.float64)*logtp
	t2 = .5*tf.squeeze(tf.matmul(tf.matmul(tf.matrix_transpose(ym),covi),ym))
	t3 = .5*covd
	t4 = .5*tf.div(tr,noise)
	
	return t1+t2+t3

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

def pairwise_distance_np(x1,x2):
	return np.transpose((x1[np.newaxis,:]-x2[:,np.newaxis])**2)

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

def pairwise_distance(x1,x2):
	# Does not currently support >1 dimensional coords (which would be axis 2 of inputs and axis 3 when adjusted)
	return tf.matrix_transpose((tf.expand_dims(x1,1)-tf.expand_dims(x2,2))**2)

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