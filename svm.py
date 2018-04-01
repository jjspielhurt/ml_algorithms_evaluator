from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
import matplotlib.pyplot as plt 

def svm(dset,percentage,kernel_type,C_type,gamma_type):
	dset=dset.values
	print(dset)
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

#run the classifier	
	clf=SVC(kernel=kernel_type,C=C_type,gamma=gamma_type)
	t0=time()
	clf.fit(features_train,labels_train)
	print("training time:",round(time()-t0,3),"s")
	t0=time()
	pred=clf.predict(features_test)
	print("predict time:",round(time()-t0,3),"s")
	print (pred)
	accuracy=accuracy_score(labels_test,pred)
	print(accuracy)
