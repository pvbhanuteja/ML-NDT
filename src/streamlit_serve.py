from __future__ import print_function
import streamlit as st
import pandas as pd
import numpy as np
import sys
# import keras
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

model_path = '/home/azureuser/ML-NDT/src/modelcpnteadc0cc9-884d-49eb-b291-af7dda4dc230.hdf5'

st.title('Augmented Ultrasonic Data for Machine Learning')

# width = st.sidebar.slider("plot width", 0.1, 25., 3.)
# height = st.sidebar.slider("plot height", 0.1, 25., 1.)


fig = plt.figure(figsize=(5,5)) 

option = st.selectbox(
     'Select the image set to predict the defect.',
     ('NONE',
    'F68B8BC9-C4D5-4848-923E-A68176F821D2.bins', 
     'FA4DC2D8-C0D9-4ECB-A319-70F156E3AF31.bins'))

if option != 'NONE':
    data_path  = f'/home/azureuser/ML-NDT/data/validation/{option}'

    st.write('You selected:', option)
    st.write(data_path)
    st.write('Starting predictions')


    model = keras.models.load_model(model_path)
    rxs = np.fromfile(data_path, dtype=np.uint16 ).astype('float32')
    rxs -= rxs.mean()
    rxs /= rxs.std()+0.0001
    rxs = np.reshape( rxs, (-1,256,256,1), 'C')

    rys = np.loadtxt(data_path.split('.')[0]+".labels", dtype=np.float32)

    predictions = model.predict(rxs)
    res = np.concatenate( (rys,predictions), -1 )
    plt.plot(res[:,1], res[:,2], 'bo')

    st.subheader('POD curve')
    st.pyplot(fig)

    st.subheader("Predictions")


    with st.spinner('Predicting 100 images'):
        cols = st.columns([5, 1,1,1])
        cols[0].write(f'Image')
        cols[1].write(f'Prediction prob')
        cols[2].write(f'True label')
        cols[3].write('Flaw size')
        

        
        tl =  rxs.shape[0]
        for i in range(0, tl):
            fig = plt.figure(figsize=(5,3))
            plt.imshow(rxs[i], cmap='viridis', interpolation='nearest')
            plt.axis('off')
            cols = st.columns([5, 1,1,1])
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.clf()
            cols[0] = st.image(buf)
            cols[1].write(f'{predictions[i][0]}')
            cols[2].write(f'{rys[i][0]}')
            cols[3].write(f'{rys[i][1]}')
