import numpy as np
import tensorflow as tf
import itertools

def svgd_kernel(X0):
	XY = tf.matmul(X0, tf.transpose(X0))
	X2_ = tf.reduce_sum(tf.square(X0), axis=1)

	x2 = tf.reshape( X2_, shape=( tf.shape(X0)[0], 1) )
	
	X2e = tf.tile(x2, [1, tf.shape(X0)[0] ] )
	H = tf.subtract(tf.add(X2e, tf.transpose(X2e) ), 2 * XY)

	V = tf.reshape(H, [-1,1]) 

	# median distance
	
	h = get_median(V)
	h = tf.sqrt(0.5 * h / tf.log( tf.cast( tf.shape(X0)[0] , tf.float64) + 1.0))

	# compute the rbf kernel
	Kxy = tf.exp(-H / h ** 2 / 2.0)

	dxkxy = -tf.matmul(Kxy, X0)
	sumkxy = tf.expand_dims(tf.reduce_sum(Kxy, axis=1), 1) 
	dxkxy = tf.add(dxkxy, tf.multiply(X0, sumkxy)) / (h ** 2)

	return (Kxy, dxkxy)

def get_median(v):
	v = tf.reshape(v, [-1])
	m = v.get_shape()[0]//2
	return tf.nn.top_k(v, m).values[m-1]

def min_pip(arr):
	if np.shape(arr)[1]<2:
		return 0
	all_min = [np.amin(np.abs(pair[0]-pair[1])) for pair in itertools.combinations(arr.T,2)]
	return np.amin(all_min)

def test_case(U_z,initial_points):
	num_particles, num_latent = np.shape(initial_points)
	#range_low = np.amin(initial_points,axis=0)
	#range_high = np.amax(initial_points,axis=0)
	z  = tf.placeholder(tf.float64, [num_particles, num_latent])
	f = -U_z(z)
	prob = tf.exp(f)
	log_p_grad = tf.squeeze(tf.gradients(f, z))
	out = svgd_kernel(z) # call MAIN update
	kernel_matrix, kernel_gradients = out[0], out[1]

	grad_theta = (tf.matmul(kernel_matrix, log_p_grad) + kernel_gradients)/num_particles
	#z_np = np.random.rand(num_particles, num_latent)*(range_high-range_low)+range_low
	z_np = initial_points

	with tf.Session() as s:
		for i in range(num_iter):
			#print('starting iteration %04i'%i,end='\r')
			grad_theta_ = s.run( grad_theta, feed_dict={z: z_np } )
			z_np = z_np + lr * grad_theta_
			print('Finished Iteration %04i. LS: %.2f-%.2f, Vr: %.2f-%.2f, Noise: %.2f-%.2f, Closest Pips: %.2f' %
				 (i,np.amin(z_np[:,0]),np.amax(z_np[:,0]),np.amin(z_np[:,1]),
				  np.amax(z_np[:,1]),np.amin(z_np[:,2]),np.amax(z_np[:,2]),
				  min_pip(z_np[:,3:])),end='\r')
			#z_np[z_np<0] = .01

	return z_np

def adagrad_train(U_z,initial_points,dim,lr=1e-3,max_iter=5000,epsilon=1e-5):
	
	num_particles, num_latent = np.shape(initial_points)
	z  = tf.placeholder(tf.float64, [num_particles, num_latent])
	f = -U_z(z)
	prob = tf.exp(f)
	log_p_grad = tf.squeeze(tf.gradients(f, z))
	if num_particles>1:
		out = svgd_kernel(z) # call MAIN update
		kernel_matrix, kernel_gradients = out[0], out[1]
		grad_theta = (tf.matmul(kernel_matrix, log_p_grad) + kernel_gradients)/num_particles
	else:
		grad_theta = log_p_grad
		
	z_np = initial_points
	
	historical_grad = 0
	historical_f = 0
	reset_count = 0
	fudge_factor = 1e-6
	alpha = 0.9

	with tf.Session() as s:
		for i in range(max_iter):
			grad_theta_, f_ = s.run([grad_theta,f], feed_dict={z: z_np })
			
			if i==0:
				historical_grad = historical_grad + grad_theta_ ** 2
			else:
				historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta_ ** 2)
			adj_grad = np.divide(grad_theta_, fudge_factor+np.sqrt(historical_grad))
			
			z_np = z_np + lr * adj_grad
			
			print('Finished Iteration %04i. LS: %.2f-%.2f, Vr: %.2f-%.2f, Noise: %.2f-%.2f' %
				 (i,np.amin(z_np[:,:dim]),np.amax(z_np[:,:dim]),
				 	np.amin(z_np[:,1]),np.amax(z_np[:,1]),
				 	np.amin(z_np[:,2]),np.amax(z_np[:,2])),end='\r')
				 	#min_pip(z_np[:,3:])),end='\r')
			
			worse_ps = f_<(historical_f-2)
			reset_count = reset_count + np.sum(worse_ps)
			if (i>0):
				for j in range(dim+2):
					# negative hps & worsening pts -> median values
					z_np[np.logical_or((z_np[:,j]<0),worse_ps),j] = np.median(z_np[:,j]) 

			historical_f = f_
			
			if (lr*np.sum(np.abs(adj_grad)))<epsilon:
				break

	return z_np, reset_count

def tf_eval(fn,fn_input):
	x = tf.placeholder(tf.float64, np.shape(fn_input))
	y = fn(x)
	with tf.Session() as s:
		out = s.run(y,feed_dict={x: fn_input})
	return out