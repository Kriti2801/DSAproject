import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import math
from sklearn.decomposition import PCA
# import sys
# np.set_printoptions(threshold=sys.maxsize)


data = pd.read_csv('HTRU2/HTRU_2.csv',sep=',',header=0)
# print data.loc[0,:]
y = data["class"]
# print data.shape
r,c = data.shape
# print r,c
X = np.zeros((r,c-1))
cols = data.columns.tolist()
# print cols
#['m_ip', 'sd_ip', 'ek_ip', 'sk_ip', 'm_dmsnr', 'sd_dmsnr', 'ek_dmsnr', 'sk_dmsnr', 'class']
X[:,0] = data[cols[0]]
X[:,1] = data[cols[1]]
X[:,2] = data[cols[2]]
X[:,3] = data[cols[3]]
X[:,4] = data[cols[4]]
X[:,5] = data[cols[5]]
X[:,6] = data[cols[6]]
X[:,7] = data[cols[7]]

X = StandardScaler().fit_transform(X)

mbk = MiniBatchKMeans(n_clusters = 2,max_iter=200,init='k-means++', n_init = 1, random_state=1)
mbk.fit(X)
centers = mbk.cluster_centers_

# #this will tell us to which cluster does the data observations belong.
new_labels = mbk.labels_
# print len(new_labels)
# print len(y.tolist())

newlabels = np.array(new_labels)
# newlabels[newlabels==1]=2
# newlabels[newlabels==0]=1
# newlabels[newlabels==2]=0
y_vals = np.array(y.tolist())
corr_preds_tot = np.sum(newlabels==y_vals)
print "Total number of data points: ",len(newlabels)
print "Total number of correct predictions: ",corr_preds_tot #in how many places both arrays have same elements
print "Overall Accuracy = ",corr_preds_tot*100.0/len(newlabels)
count = np.zeros((2),dtype=int)
for i in range(len(newlabels)):
	if y_vals[i]==0 and y_vals[i]==newlabels[i]:
		count[0] = count[0]+1
	if y_vals[i]==1 and y_vals[i]==newlabels[i]:
		count[1] = count[1]+1
# print count

print "Correctly predicted 0s: ",count[0]," out of 16,259"
print "Accuracy of 0s prediction: ", count[0]*100.0/16259
print "Correctly predicted 1s: ",count[1]," out of 1,639"
print "Accuracy of 1s prediction: ", count[1]*100.0/1639