import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.decomposition import PCA
# import sys
# np.set_printoptions(threshold=sys.maxsize)

data = pd.read_csv('HTRU2/HTRU_2.csv',sep=',',header=0)
# print data.loc[0,:]
y = data["class"] #y=np.array(data)[:,-1]
# print data.shape
r,c = data.shape
# print r,c
X = np.array(data)[:,:-1]

X = StandardScaler().fit_transform(X)


# i=2
for i in range(2,c-1):
	pca = PCA(n_components=i)
	print "PCA components = ",i,"\n"
	principalComponents = pca.fit_transform(X)
	# print(principalComponents.shape)

	affinities = ["euclidean", "l1", "l2", "manhattan", "cosine"]
	linkages = ["ward", "complete", "average", "single"]
	hc = AgglomerativeClustering(n_clusters = 2, affinity = affinities[0], linkage = linkages[0])
	y_hc = hc.fit_predict(principalComponents)

	y_vals = np.array(y.tolist())

	corr_preds_tot = np.sum(y_hc==y_vals)
	over_acc = corr_preds_tot*100.0/len(y_hc)

	if over_acc<50.0:
		y_hc[y_hc==1] = 2
		y_hc[y_hc==0] = 1
		y_hc[y_hc==2] = 0

	corr_preds_tot = np.sum(y_hc==y_vals)
	over_acc = corr_preds_tot*100.0/len(y_hc)
	count = np.zeros((2),dtype=int)
	for i in range(len(y_hc)):
		if y_vals[i]==0 and y_vals[i]==y_hc[i]:
			count[0] = count[0]+1
		if y_vals[i]==1 and y_vals[i]==y_hc[i]:
			count[1] = count[1]+1
	# print count
	print "Total 0s = ",count[0],"\t","Total 1s = ",count[1]
	acc_0s = count[0]*100.0/16259
	acc_1s = count[1]*100.0/1639
	print "Accuracy of 0s prediction: ",acc_0s
	print "Accuracy of 1s prediction: ",acc_1s
	print "Overall Accuracy = ",over_acc
	print "---------------------------------------------------"
