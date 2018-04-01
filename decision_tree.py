from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time

def tree(features_train,labels_train,features_test,labels_test,max_depth_type,min_samples_split_type,min_samples_leaf_type):
	clf=DecisionTreeClassifier(max_depth=max_depth_type,min_samples_split=min_samples_split_type,min_samples_leaf=min_samples_leaf_type)
	t0=time()
	clf.fit(features_train,labels_train)
	print("training time:",round(time()-t0,3),"s")
	t0=time()
	pred=clf.predict(features_test)
	print("predict time:",round(time()-t0,3),"s")
	print (pred)
	accuracy=accuracy_score(labels_test,pred)
	print(accuracy)
