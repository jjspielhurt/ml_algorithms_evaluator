import pandas as pd
def r_dataset():

	file_dir=input("Path to the dataset:")
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
dset=r_dataset()

#linear regression
if alg_type=='1' :
	repeat='y'
	while(repeat=='y'):
		from linear_regression import linear_regression

		learning_rate=float(input("Learning rate:"))
		num_iterations=int(input("Number of iterations:"))
		initial_m=float(input("Initial m:"))
		initial_b=float(input("Initial b:"))
		linear_regression(dset,learning_rate,num_iterations,initial_m,initial_b)

		repeat=input("Do it again with perhaps other paramethers?(y/n):")

		while not (repeat == 'y' or repeat == 'n'):
			repeat=input("y/n:")


if alg_type=='2':
	repeat='y'
	while(repeat=='y'):
		print("1.Naive Bayes 2.SVM 3.Decision Tree")
		c_type=input("1/2/3:\n")
		while not (c_type == '1' or c_type == '2' or c_type == '3'):
			print("You entered an invalid input.Please try again (1, 2 or 3):") 
			c_type=input("1/2/3:\n")

		percentage=float(input("What percentage of the dataset shall we use for training?:"))
		while percentage>1 or percentage<0:
			percentage=float(input("Error.Please use a number between 0 and 1:"))
		
		if not('numpy.ndarray' in str(type(dset)) ):
			dset=dset.values
		features_train=[]
		features_test=[]
		labels_train=[]
		labels_test=[]
		
	#extract training data
		for row in dset[:int(len(dset)*percentage)]:
			feature_row=[]
			for feature_i in range(0,len(row)-1):
				feature_row.append(row[feature_i])
			features_train.append(feature_row)
		for row in dset[:int(len(dset)*percentage)]	:
			labels_train.append(row[len(row)-1])
	#extract testing data
		for row in dset[int(len(dset)*percentage):]:
			feature_row=[]
			for feature_i in range(0,len(row)-1):
				feature_row.append(row[feature_i])
			features_test.append(feature_row)
		for row in dset[int(len(dset)*percentage):]	:
			labels_test.append(row[len(row)-1])

			
	#Naive Bayes
		if c_type=='1':
			from naive_bayes import nb
			nb(features_train,labels_train,features_test,labels_test)
	#Support Vector Machine
		if c_type=='2':
			from svm import svm
			kernel_type=input("Kernel:")
			C_type=float(input("C:"))
			gamma_type=float(input("Gamma:"))
			svm(features_train,labels_train,features_test,labels_test,kernel_type,C_type,gamma_type)

	#Decision Tree
		if c_type=='3':
			from decision_tree import tree
			max_depth_type=int(input("Maximum tree depth:"))
			min_samples_split_type=float(input("Minimum samples required to split(min_samples_split):"))
			#required by classifier
			if(min_samples_split_type>=1):
				min_samples_split_type=int(min_samples_split_type)

			min_samples_leaf_type=float(input("Minimum samples in one leaf(min_samples_leaf):"))
			#required by classifier
			if(min_samples_leaf_type>=1):
				min_samples_leaf_type=int(min_samples_leaf_type)

			tree(features_train,labels_train,features_test,labels_test,max_depth_type,min_samples_split_type,min_samples_leaf_type)

		repeat=input("Do it again with perhaps other paramethers or classifiers?(y/n):")

		while not (repeat == 'y' or repeat == 'n'):
			repeat=input("y/n:")
