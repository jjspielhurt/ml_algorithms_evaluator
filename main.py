import pandas as pd
def r_dataset():

	file_dir=input("Path to the dataset:\n *2 features,each point on one line of the file ")
	valid=False
	while valid==False:
		try:
			pd.read_table(file_dir,delimiter=" ",header=None)
			valid=True
		except:
			file_dir=input("No file found at this path.Try again:")
			valid=False
	dset=pd.read_table(file_dir,delimiter=" ",header=None)
	return dset


print("Hi!What type of machine learning algorithm you need?")
print("1.Linear Regression  2.Classificasion")
alg_type=input("1/2:\n")

while not (alg_type == '1' or alg_type == '2'):
	print("You entered an invalid input.Please try again (1 or 2):") 
	alg_type=input("1/2:\n")
#linear regression
if alg_type=='1' :
	from linear_regression import linear_regression

	dset=r_dataset()

	learning_rate=float(input("Learning rate:"))
	num_iterations=int(input("Number of iterations:"))
	initial_m=float(input("Initial m:"))
	initial_b=float(input("Initial b:"))
	linear_regression(dset,learning_rate,num_iterations,initial_m,initial_b)
if alg_type=='2':
	print("1.Naive Bayes 2.SVM 3.Decision Tree")
	c_type=input("1/2/3:\n")
	while not (c_type == '1' or c_type == '2' or c_type == '3'):
		print("You entered an invalid input.Please try again (1, 2 or 3):") 
		c_type=input("1/2/3:\n")

	dset=r_dataset()
	percentage=float(input("What percentage of the dataset shall we use for training?"))
	while percentage>1 or percentage<0:
		percentage=float(input("Error.Please use a number between 0 and 1:"))
		
#Naive Bayes
	if c_type=='1':
		from naive_bayes import nb
#Support Vector Machine
	if c_type=='2':
		from svm import svm
		kernel_type=input("Kernel:")
		C_type=float(input("C:"))
		gamma_type=float(input("Gamma:"))
		svm(dset,percentage,kernel_type,C_type,gamma_type)

#Decision Tree
	if c_type=='3':
		from decision_tree import tree

