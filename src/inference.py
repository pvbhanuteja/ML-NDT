
from __future__ import print_function
import sys
# import keras
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

model_path = '/home/azureuser/ML-NDT/src/modelcpnteadc0cc9-884d-49eb-b291-af7dda4dc230.hdf5'
data_path  = '/home/azureuser/ML-NDT/data/validation/F68B8BC9-C4D5-4848-923E-A68176F821D2.bins'

model = keras.models.load_model(model_path)
rxs = np.fromfile(data_path, dtype=np.uint16 ).astype('float32')
rxs -= rxs.mean()
rxs /= rxs.std()+0.0001
rxs = np.reshape( rxs, (-1,256,256,1), 'C')

predictions = model.predict(rxs)
print(predictions)
