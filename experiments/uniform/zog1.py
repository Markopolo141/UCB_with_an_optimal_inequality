from random import random, betavariate, uniform
from math import sqrt,log
from tqdm import tqdm
from operator import mul
import numpy

# add d to v, making sure that the result is atleast greater than v
minimum_add = lambda v,d : max(v+d,numpy.nextafter(v,float("inf")))

# search for a solution of monotonic increasing function f(x)=value, from x=lower, past x=upper>lower
def open_search(f,upper,lower,value,iterations,args):
	if f(lower,args)>= value:
		return lower
	v = lower
	diff = upper-lower
	for n in range(1,iterations+1):
		d = diff*1.0/(2**n)
		while (f(minimum_add(v,d),args)<=value):
			v = minimum_add(v,d)
	return v

# the function defining appropriate G value for pmax.
def Gfunction(G,args):
	u = args[0]
	s = args[1:]
	return sum([G*(ss-u)/(ss-u-G) for ss in s])

# returns negative of pmax, for an appropriate g
def minus_pmax(u, s):
	uhat = sum(s)*1.0/len(s)
	if (uhat>=u):
		return -1.0
	if (1.0 not in s) and (Gfunction(1.0-u,[u]+s) >= 0): #if G at the boundaries is valud
		G = 1.0-u
	else: #otherwise search for a solution G for Gfunction=0
		G = open_search(Gfunction,2.0**6-u,minimum_add(1.0-u,0),0,25,[u]+s)
	return -reduce(mul,[G/(G-ss+u) for ss in s])

#classic UCB1 formulation, returns index of selected arm
def UCB1(samples,t):
	for i in range(len(samples)):
		if len(samples[i]) == 0:
			return i
	sample_lengths = [len(s) for s in samples]
	sample_means = [sum(s)*1.0/len(s) for s in samples]
	bounds = [sample_means[i] + sqrt(2*log(t)/(l)) for i,l in enumerate(sample_lengths)]
	return bounds.index(max(bounds))

#our UCBOB formulation, returns index of selected arm
def UCBOB(samples,t):
	for i in range(len(samples)):
		if len(samples[i]) == 0:
			return i
	sample_means = [sum(s)*1.0/len(s) for s in samples]
	upper_bounds = []
	for i in range(len(samples)):
		v = open_search(minus_pmax,1,sample_means[i],-1.0/(t*(log(t+0.01)**3)),30,samples[i])
		upper_bounds.append(v)
	return upper_bounds.index(max(upper_bounds))

def UCBV(samples,t):
	for i in range(len(samples)):
		if len(samples[i]) == 0:
			return i
	sample_lengths = [len(s) for s in samples]
	sample_means = [sum(s)*1.0/len(s) for s in samples]
	sample_variances = [sum([(ss-sample_means[i])**2 for ss in s])*1.0/sample_lengths[i] for i,s in enumerate(samples)]
	bounds = [sample_means[i]+sqrt(sample_variances[i]*log(t)*2.0/sample_lengths[i])+sqrt(log(t)*1.0/sample_lengths[i]) for i,l in enumerate(sample_lengths)]
	return bounds.index(max(bounds))

def RAND(samples,t):
	return int(random()*len(samples)*0.999)


K = 5
max_trials = 100
max_time = 500
methods = [UCB1,UCBOB,UCBV,RAND]

trial_data = [ [] for i in range(len(methods))]
for method_i,method in enumerate(methods): #for each method
	print "running {}".format(str(method))
	for trial in tqdm(range(max_trials)):  #for each possible trial run
		trial_data[method_i].append([])
		a_k = [random() for k in range(K)] #generate a bandit problem
		b_k = [random() for k in range(K)]
		means = [(a_k[i]+b_k[i])*1.0/2 for i in range(K)]
		max_mean = max(means)
		samples = [[] for i in range(K)]
		len_samples = [0 for i in range(K)]
		for t in range(1,max_time+1): #run the method
			arm = method(samples,t)
			samples[arm].append(uniform(a_k[arm],b_k[arm]))
			len_samples[arm] += 1
			trial_data[method_i][-1].append(sum([len_samples[i]*(max_mean-means[i]) for i in range(K)]))

# transpose data arrays and calculate average regrets
rearranged_trial_data = [list(map(list, zip(*l))) for l in trial_data]
for i in range(len(rearranged_trial_data)):
	for j in range(len(rearranged_trial_data[i])):
		vv = rearranged_trial_data[i][j]
		rearranged_trial_data[i][j] = sum(vv)*1.0/len(vv)

#output
with open("data22.csv","w") as f:
	for r in rearranged_trial_data:
		f.write(",".join([str(rr) for rr in r]))
		f.write("\n")
with open("data22.tikz","w") as f:
	for r in rearranged_trial_data:
		for i,rr in enumerate(r):
			f.write("({},{})".format(i,rr))
		f.write("\n\n")
