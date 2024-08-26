import ipaddress
from ipaddress import IPv4Address, IPv4Network
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from sklearn import svm


# ------------------------------------------------------------
# ------------------------------------------------------------
df = pd.ExcelFile('Hacker_Data_BaseN.xlsx').parse('Patterns')
address = df['IP_Address']
address_unique = address.drop_duplicates().tolist()
address_ip_unique = []
for i in range(len(address_unique)):
    parts = address_unique[i].split('.')
    address_ip_unique.append(parts)
# data['Name'] = data['Name'].str.upper()
address_train = np.array(address_ip_unique)

classA = ['0.0.0.0', '127.255.255.255']
classB = ['128.0.0.0', '191.255.255.255']
classC = ['192.0.0.0', '223.255.255.255']
classD = ['224.0.0.0', '239.255.255.255']
classE = ['240.0.0.0', '255.255.255.254']

for i in range(len(address_unique)):
    ip = IPv4Address(address_unique[i])
    if IPv4Address(classA[0]) < ip < IPv4Address(classA[1]):
        print(ip, ' ----> Class A')
    elif IPv4Address(classB[0]) < ip < IPv4Address(classB[1]):
        print(ip, ' ----> Class B')
    elif IPv4Address(classC[0]) < ip < IPv4Address(classC[1]):
        print(ip, ' ----> Class C')
    elif IPv4Address(classD[0]) < ip < IPv4Address(classD[1]):
        print(ip, ' ----> Class D')
    elif IPv4Address(classE[0]) < ip < IPv4Address(classE[1]):
        print(ip, ' ----> Class E')
    else:
        print(ip, ' ----> Class Unspecified')
'''
    ip = ipaddress.IPv4Address(address_unique[i])

    # Print total number of bits in the ip.
    print("Total no of bits in the ip:", ip.max_prefixlen)

    # Print True if the IP address is reserved for multicast use.
    print("Is multicast:", ip.is_multicast)

    # Print True if the IP address is allocated for private networks.
    print("Is private:", ip.is_private)

    # Print True if the IP address is global.
    print("Is global:", ip.is_global)

    # Print True if the IP address is unspecified.
    print("Is unspecified:", ip.is_unspecified)

    # Print True if the IP address is otherwise IETF reserved.
    print("Is reversed:", ip.is_reserved)
    
    # Print True if the IP address is a loopback address.
    print("Is loopback:", ip.is_loopback)

    # Print True if the IP address is Link-local
    print("Is link-local:", ip.is_link_local)
    
dataNorm = StandardScaler().fit_transform(address_train)
# data = address_train[:400]
pca = PCA(n_components=2)
dataPCA = pca.fit_transform(dataNorm)
data = dataPCA[:400]
print(len(dataPCA))
plt.figure()
plt.plot(dataPCA[:, 0], dataPCA[:, 1], 'og')
plt.show()

bgms = BayesianGaussianMixture(n_components=3, n_init=1, random_state=100)
bgms.fit(data)
print(np.round(bgms.weights_, 2), bgms.means_)

pca_means = pca.transform(bgms.means_)
print(bgms.means_)
print(pca_means)

npredict = bgms.predict(dataPCA[400:])
print(npredict)

print('ONE CLASS SVM')
clf = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma='scale')
clf.fit(data)
y_predict = clf.predict(dataPCA[400:])
y_predict_n = 1 - ((y_predict + 1) // 2)
print(y_predict_n)
'''
