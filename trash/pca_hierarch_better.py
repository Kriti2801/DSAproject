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

affinities = ["euclidean", "l1", "l2", "manhattan", "cosine"]
linkages = ["ward", "complete", "average", "single"]
max_acc = [0.0,"put_name_here"]
max_0s_acc = [0.0,"put_name_here"]
max_1s_acc = [0.0,"put_name_here"]
accs = np.zeros((6,4,5))
for link in range(len(linkages)):
	for affs in range(len(affinities)):
		if link==0 and affs>0:
			continue
		# print i,j
		print "Linkage,Affinity = ",linkages[link],",",affinities[affs],"\n"

		for ind_pca in range(2,c-1):
			pca = PCA(n_components=ind_pca)
			print "PCA components = ",ind_pca,"\n"
			principalComponents = pca.fit_transform(X)
			hc = AgglomerativeClustering(n_clusters = 2, affinity = affinities[affs], linkage = linkages[link])
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
			print "\tTotal number of data points: ",len(y_hc)
			print "\tTotal number of correct predictions: ",corr_preds_tot #in how many places both arrays have same elements
			print "\tOverall Accuracy = ",over_acc
			count = np.zeros((2),dtype=int)
			for i in range(len(y_hc)):
				if y_vals[i]==0 and y_vals[i]==y_hc[i]:
					count[0] = count[0]+1
				if y_vals[i]==1 and y_vals[i]==y_hc[i]:
					count[1] = count[1]+1
			# print count
			acc_0s = count[0]*100.0/16259
			acc_1s = count[1]*100.0/1639
			print "\tCorrectly predicted 0s: ",count[0]," out of 16,259"
			print "\tAccuracy of 0s prediction: ", acc_0s
			print "\tCorrectly predicted 1s: ",count[1]," out of 1,639"
			print "\tAccuracy of 1s prediction: ", acc_1s
			print "\n\n"
			accs[ind_pca-2,link,affs] = over_acc
			if over_acc>max_acc[0]:
				max_acc[0] = over_acc
				max_acc[1] = linkages[link]+","+affinities[affs]+","+"PCA comps: "+str(ind_pca)

			if acc_0s>max_0s_acc[0]:
				max_0s_acc[0] = acc_0s
				max_0s_acc[1] = linkages[link]+","+affinities[affs]+","+"PCA comps: "+str(ind_pca)

			if acc_1s>max_1s_acc[0]:
				max_1s_acc[0] = acc_1s
				max_1s_acc[1] = linkages[link]+","+affinities[affs]+","+"PCA comps: "+str(ind_pca)
			print "--------End of PCA iter--------"
		print "-----------------------------------------------------------"


# max_acc_inds = np.argwhere(accs==np.max(accs))
# # print "Max Accuracy = ",max_acc[0], "In ", max_acc[1]
# print "Max Accuracy = ",np.max(accs), "In:"
# for i in range(len(max_acc_inds)):
# 	print linkages[max_acc_inds[i,0]]+","+affinities[max_acc_inds[i,1]]

print accs
print "Max 0s Accuracy = ",max_0s_acc[0], "In ", max_0s_acc[1]
print "Max 1s Accuracy = ",max_1s_acc[0], "In ", max_1s_acc[1]
