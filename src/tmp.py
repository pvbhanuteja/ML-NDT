import numpy as np
import matplotlib.pyplot as plt

path = "../data/training/"          #training data path
vpath = "../data/validation/"       #validation data path


test_uuid = "FA4DC2D8-C0D9-4ECB-A319-70F156E3AF31"
rxs = np.fromfile(vpath+test_uuid+".bins", dtype=np.uint16 ).astype('float32')
rxs -= rxs.mean()
rxs /= rxs.std()+0.0001
rxs = np.reshape( rxs, (-1,256,256,1), 'C')
rys = np.loadtxt(vpath+test_uuid+".labels", dtype=np.float32)

print(rxs.shape,rys.shape)
i =50


plt.imshow(rxs[i], cmap='viridis', interpolation='nearest')
plt.show()
plt.savefig("tmp.png")
k = 0
for j in rys:
    
    print(k,'\n',rys[k])
    k += 1