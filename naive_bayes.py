from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time

def nb(features_train,labels_train,features_test,labels_test):
	clf=GaussianNB()
	t0=time()
	clf.fit(features_train,labels_train)
	print("training time:",round(time()-t0,3),"s")
	t0=time()
	pred=clf.predict(features_test)
	print("predict time:",round(time()-t0,3),"s")
	print (pred)
	accuracy=accuracy_score(labels_test,pred)
	print(accuracy)
