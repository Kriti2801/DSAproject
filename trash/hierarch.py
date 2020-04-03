import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
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

dendrogrm = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Pulsars')
plt.ylabel('Euclidean distance')
plt.show()

affinities = ["euclidean", "l1", "l2", "manhattan", "cosine"]
linkages = ["ward", "complete", "average", "single"]
hc = AgglomerativeClustering(n_clusters = 2, affinity = affinities[0], linkage = linkages[1])
y_hc = hc.fit_predict(X)
# print type(y_hc)
# print "done"
# print y_hc
# y_hc[y_hc==1] = 2
# y_hc[y_hc==0] = 1
# y_hc[y_hc==2] = 0
# print len(y_hc[y_hc==0])
# print len(y_hc[y_hc==1])


y_vals = np.array(y.tolist())
corr_preds_tot = np.sum(y_hc==y_vals)
# print corr_preds_tot


print "Total number of data points: ",len(y_hc)
print "Total number of correct predictions: ",corr_preds_tot #in how many places both arrays have same elements
print "Overall Accuracy = ",corr_preds_tot*100.0/len(y_hc)
count = np.zeros((2),dtype=int)
for i in range(len(y_hc)):
	if y_vals[i]==0 and y_vals[i]==y_hc[i]:
		count[0] = count[0]+1
	if y_vals[i]==1 and y_vals[i]==y_hc[i]:
		count[1] = count[1]+1
# print count

print "Correctly predicted 0s: ",count[0]," out of 16,259"
print "Accuracy of 0s prediction: ", count[0]*100.0/16259
print "Correctly predicted 1s: ",count[1]," out of 1,639"
print "Accuracy of 1s prediction: ", count[1]*100.0/1639

print "\nMetrics are:",metrics.homogeneity_completeness_v_measure(y_vals,y_hc)