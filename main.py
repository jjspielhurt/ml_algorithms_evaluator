import pandas as pd

print("Hi!What type of machine learning algorithm you need?")
print("1.Regression  2.Classificasion")
t=input("1/2:\n")

while not (t == '1' or t == '2'):
	print("You entered an invalid input.Please try again (1 or 2):") 
	t=input("1/2:\n")

if t=='1' :
	from linear_regression import linear_regression
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
	learning_rate=float(input("Learning rate:"))
	num_iterations=int(input("Number of iterations:"))
	initial_m=float(input("Initial m:"))
	initial_b=float(input("Initial b:"))
	linear_regression(dset,learning_rate,num_iterations,initial_m,initial_b)
