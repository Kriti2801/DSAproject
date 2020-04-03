import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import math
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
print "Total number of data points: ",len(y),"\n"
iters = 1
km_over_acc = np.zeros(iters)
km_0s_acc = np.zeros(iters)
km_1s_acc = np.zeros(iters)

mbk_over_acc = np.zeros(iters)
mbk_0s_acc = np.zeros(iters)
mbk_1s_acc = np.zeros(iters)

# print np.shape(km_over_acc)
for i in range(iters):
	print "Iteration = %d" %i," :\n"
	km = KMeans(n_clusters = 2,max_iter=200,init='k-means++',n_init=1, n_jobs = 4, random_state=i)
	km.fit(X)
	centers = km.cluster_centers_
	new_labels_km = km.labels_
	newlabels_km = np.array(new_labels_km)
	
	y_vals = np.array(y.tolist())
	corr_preds_tot_km = np.sum(newlabels_km==y_vals)
	km_over_acc[i] = corr_preds_tot_km*100.0/len(newlabels_km)
	if km_over_acc[i]<50.0:
		newlabels_km[newlabels_km==1]=2
		newlabels_km[newlabels_km==0]=1
		newlabels_km[newlabels_km==2]=0
	corr_preds_tot_km = np.sum(newlabels_km==y_vals)
	km_over_acc[i] = corr_preds_tot_km*100.0/len(newlabels_km)
	
	# print "Results of KMeans: "
	# print "\tTotal number of correct predictions by : ",corr_preds_tot_km #in how many places both arrays have same elements
	# print "\tOverall Accuracy = ",km_over_acc[i]
	count_km = np.zeros((2),dtype=int)
	for p in range(len(newlabels_km)):
		if y_vals[p]==0 and y_vals[p]==newlabels_km[p]:
			count_km[0] = count_km[0]+1
		if y_vals[p]==1 and y_vals[p]==newlabels_km[p]:
			count_km[1] = count_km[1]+1
	# print count_km
	km_0s_acc[i] = count_km[0]*100.0/16259
	km_1s_acc[i] = count_km[1]*100.0/1639

	# print "\tCorrectly predicted 0s: ",count_km[0]," out of 16,259"
	# print "\tAccuracy of 0s prediction: ", km_0s_acc[i]
	# print "\tCorrectly predicted 1s: ",count_km[1]," out of 1,639"
	# print "\tAccuracy of 1s prediction: ", km_1s_acc[i]
	# print "\n"

	mbk = MiniBatchKMeans(n_clusters = 2,max_iter=200,init='k-means++', n_init = 1, random_state=1)
	mbk.fit(X)
	centers = mbk.cluster_centers_
	new_labels_mbk = mbk.labels_
	newlabels_mbk = np.array(new_labels_mbk)
	# y_vals = np.array(y.tolist())
	corr_preds_tot_mbk = np.sum(newlabels_mbk==y_vals)
	mbk_over_acc[i] = corr_preds_tot_mbk*100.0/len(newlabels_mbk)
	if mbk_over_acc[i]<50.0:
		newlabels_mbk[newlabels_mbk==1]=2
		newlabels_mbk[newlabels_mbk==0]=1
		newlabels_mbk[newlabels_mbk==2]=0
	corr_preds_tot_mbk = np.sum(newlabels_mbk==y_vals)
	mbk_over_acc[i] = corr_preds_tot_mbk*100.0/len(newlabels_mbk)
	# print "Results of MiniBatchKMeans: "
	# print "\tTotal number of data points: ",len(newlabels_mbk)
	# print "\tTotal number of correct predictions: ",corr_preds_tot_mbk #in how many places both arrays have same elements
	# print "\tOverall Accuracy = ",mbk_over_acc[i]
	count_mbk = np.zeros((2),dtype=int)
	for q in range(len(newlabels_mbk)):
		if y_vals[q]==0 and y_vals[q]==newlabels_mbk[q]:
			count_mbk[0] = count_mbk[0]+1
		if y_vals[q]==1 and y_vals[q]==newlabels_mbk[q]:
			count_mbk[1] = count_mbk[1]+1
	# print count_mbk

	mbk_0s_acc[i] = count_mbk[0]*100.0/16259
	mbk_1s_acc[i] = count_mbk[1]*100.0/1639
	# print "\tCorrectly predicted 0s: ",count_mbk[0]," out of 16,259"
	# print "\tAccuracy of 0s prediction: ", mbk_0s_acc[i]
	# print "\tCorrectly predicted 1s: ",count_mbk[1]," out of 1,639"
	# print "\tAccuracy of 1s prediction: ", mbk_1s_acc[i]