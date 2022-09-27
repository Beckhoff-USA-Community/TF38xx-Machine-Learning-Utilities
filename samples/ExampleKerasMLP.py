################################################################
# Example - Train and export a Keras MLP                       #
#                                                              #
# Framework: Keras with Tensorflow backend                     #
# Train and test data are sampled from cos(x)+gaussian noise   #
# so after training the network should have learned            #
# to predict 2*cos(x)+0.5*rand, where -pi<=x<=pi.              #
#                                                              #
# About the model:                                             #
# Number of hidden layer: 1                                    #
# Number of inputs: 1                                          #
# Number of outpus: 1                                          #
# Activations: tanh (hidden layer), linear (output)            #
# Optimizer: Adamax                                            #
################################################################

################################################################
# Import                                                       #
################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_WARNINGS"] = "FALSE"
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adamax
from KerasMlp2Xml import net2xml

################################################################
# Set Parameters                                               #
################################################################
num_samples_train = 3000 # number of train samples
num_samples_test = 1500 # number of test samples
sample_min = -np.pi # min(samples)
sample_max = np.pi # max(samples)
noise_factor = 0.1

learning_rate = 0.001
loss = 'mean_squared_error'
batch_size = 64
num_epochs = 100

################################################################
# Generate Train And Test Data                                 #
################################################################
# Generate input train data (random)
data_x_train = np.transpose((sample_max - sample_min)*np.random.rand(1,num_samples_train) + sample_min)
# Sort train data (only for plotting)
data_x_train_sorted = np.sort(data_x_train,axis=0)
# Calculate output train data with gaussian noise
data_y_train = 2*np.cos(data_x_train) + 0.5 + noise_factor*np.random.normal(loc=0,scale=1,size=np.shape(data_x_train))
# Output scaling, e.g. normalization
data_y_train_mean = np.mean(data_y_train)
data_y_train_std = np.std(data_y_train)
output_scaling_bias = data_y_train_mean
output_scaling_scal = data_y_train_std
data_y_train_scal = (1/output_scaling_scal)*(data_y_train - output_scaling_bias)

# Generate test data (random)
data_x_test = np.transpose((sample_max - sample_min)*np.random.rand(1,num_samples_test) + sample_min)
# Sort test data (only for plotting)
data_x_test_sorted = np.sort(data_x_test,axis=0)
# Calculate output test data with gaussian noise
data_y_test = 2*np.cos(data_x_test) + 0.5 + noise_factor*np.random.normal(loc=0,scale=1,size=np.shape(data_x_test))
# Output scaling
data_y_test_scal = (1/output_scaling_scal)*(data_y_test-output_scaling_bias)

################################################################
# Initialize Model And Set Train Parameters                    #
################################################################
model = Sequential()
model.add(Dense(3,input_dim=1,activation='tanh'))
model.add(Dense(1,activation='linear'))
model.summary()

optimizer = Adamax(lr=learning_rate) # Set optimizer
model.compile(optimizer=optimizer, loss=loss) # configure model for training

################################################################
# Train And Plot                                               #
################################################################
figure = plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
subplot_mse = figure.add_subplot(131)
subplot_mse.grid()
subplot_mse.set_title('Model Loss')
subplot_mse.set_xlabel('Epoch')
subplot_mse.set_ylabel('MSE')

subplot_train = figure.add_subplot(132)
subplot_train.grid()
subplot_test = figure.add_subplot(133)
subplot_test.grid()

history = {'loss':[], 'val_loss':[]}
for idx_epoch in range(1,num_epochs+1):
    # training with scaled train data
    history_temp = model.fit(x=data_x_train,y=data_y_train_scal,batch_size=batch_size,epochs=1,verbose=0,validation_data=(data_x_test,data_y_test_scal),shuffle=True)

    # save loss history
    history['loss'].append(history_temp.history['loss'][0])
    history['val_loss'].append(history_temp.history['val_loss'][0])

    # plot
    subplot_mse.plot([i for i in range(1, idx_epoch + 1)], history['loss'], '-b')
    subplot_mse.plot([i for i in range(1, idx_epoch + 1)], history['val_loss'], '-g')
    subplot_mse.legend(['Train', 'Test'], loc='upper right')

    subplot_train.clear()
    subplot_train.plot(data_x_train,data_y_train_scal,'.g')
    subplot_train.plot(data_x_train_sorted,(1/output_scaling_scal)*((2*np.cos(data_x_train_sorted)+0.5) - output_scaling_bias), '-b')
    subplot_train.plot(data_x_train_sorted, model.predict(data_x_train_sorted), '-r')
    subplot_train.set_title('Model Output - Train Data (Scaled)')
    subplot_train.set_xlabel('Input')
    subplot_train.set_ylabel('Target/Output')
    subplot_train.legend(['Train Data', '2*cos+0.5', 'Output Model'], loc='upper right')

    subplot_test.clear()
    subplot_test.plot(data_x_test,data_y_test_scal,'.g')
    subplot_test.plot(data_x_test_sorted,(1/output_scaling_scal)*  ((2*np.cos(data_x_test_sorted)+0.5) - output_scaling_bias) , '-b')
    subplot_test.plot(data_x_test_sorted,model.predict(data_x_test_sorted), '-r')
    subplot_test.set_title('Model Output - Test Data (Scaled)')
    subplot_test.set_xlabel('Input')
    subplot_test.set_ylabel('Target/Output')
    subplot_test.legend(['Test Data', '2*cos+0.5', 'Output Model'], loc='upper right')

    plt.draw()
    plt.pause(0.5)

    if idx_epoch>10 and abs(max(history['loss'][-10:]) - min(history['loss'][-10:])) < 1e-4:
        # Stop training if the train loss did not change significantly in the last 10 epochs
        break

################################################################
# Export Model Parameters                                      #
################################################################
# Generate xml string

# The model was trained with scaled output data.
# In order to automatically undo the scaling of predictions further parameters (output_scaling_bias, output_scaling_scal) have to be exported.
# One should mind the order of these parameters!
# output_scaling_bias and output_scaling_scal must satisfy the following equation:
# non_scaled_predictions = output_scaling_scal * scaled_predictions + output_scaling_bias

# If no output scaling is used the parameters can be ignored because they are optional.

xml = net2xml(model, output_scaling_bias, output_scaling_scal)
# Save string in *.xml-file
with open('KerasMLPExample.xml', 'w') as file:
    file.write(xml)

plt.show()