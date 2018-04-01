from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
import matplotlib.pyplot as plt 

def svm(features_train,labels_train,features_test,labels_test,kernel_type,C_type,gamma_type):

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
