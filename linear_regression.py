import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
def step_gradient(b_current,m_current,points,learning_rate):
	#gradient descent
	b_gradient=0
	m_gradient=0
	N=float(len(points))#number of points
	for i in range(0,len(points)):
		x=points[i,0]
		y=points[i,1]
		b_gradient+=-(2/N)*(y-((m_current*x)+b_current))
		m_gradient+=-(2/N)*x*(y-((m_current*x)+b_current))
	new_b=b_current-b_gradient*learning_rate
	new_m=m_current-m_gradient*learning_rate
	return [new_b,new_m]

def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_interations):

	b=starting_b
	m=starting_m
	for i in range(num_interations):
		b,m=step_gradient(b,m,np.array(points),learning_rate)	
	return [b,m]

def linear_regression(points,learning_rate,num_interations,initial_m,initial_b):
	points=np.array(points)
	x=[]
	y=[]
	for i in range(0,len(points)):
		x.append(points[i,0])
		y.append(points[i,1])
	plt.scatter(x,y,color="b",label="train data")
	print("Starting Linear Regression")
	t0=time()
	[b,m]=gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_interations)
	print("Time:",round(time()-t0,3),"s")
	print("Predicted intercept:",b,"\n","Predicted slope:",m)
	y_predicted=[]
	for i in x:
		y_predicted.append(m*i+b)

	plt.plot(x,y_predicted,color="red")
	plt.show()
