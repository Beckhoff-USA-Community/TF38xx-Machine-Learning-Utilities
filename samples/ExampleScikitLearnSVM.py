################################################################
# Example - Train and export a Scikit Learn SVM (NuSVR)        #
#                                                              #
# Train and test data are sampled from sin(x)+gaussian noise   #
# so after training the model should have learned              #
# to predict sin(x), where -pi<=x<=pi.                         #
################################################################

################################################################
# Import                                                       #
################################################################
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
import numpy as np
from ScikitLearnSvm2Xml import svm2xml

################################################################
# Set Parameters                                               #
################################################################
num_samples_train = 500 # number of train samples
num_samples_test = 300 # number of test samples
sample_min = -np.pi # min(samples)
sample_max = np.pi # max(samples)
noise_factor = 0.1

################################################################
# Generate Train And Test Data                                 #
################################################################
# Generate input train data (random)
data_x_train = np.transpose((sample_max - sample_min)*np.random.rand(1,num_samples_train) + sample_min)
# Sort train data (only for plotting)
data_x_train_sorted = np.sort(data_x_train,axis=0)

# Input scaling, e.g. normalization
data_x_train_mean = np.mean(data_x_train)
data_x_train_std = np.std(data_x_train)
input_scaling_bias = -data_x_train_mean/data_x_train_std
input_scaling_scal = 1/data_x_train_std
data_x_train_scal = input_scaling_scal * data_x_train + input_scaling_bias
data_x_train_sorted_scal = input_scaling_scal * data_x_train_sorted + input_scaling_bias

# Calculate output train data with gaussian noise
data_y_train = np.sin(data_x_train_scal) + noise_factor*np.random.normal(loc=0,scale=1,size=np.shape(data_x_train_scal))
# Reshape output train data
data_y_train = data_y_train.reshape((data_y_train.shape[0],))

# Generate test data (random)
data_x_test = np.transpose((sample_max - sample_min)*np.random.rand(1,num_samples_test) + sample_min)
# Input scaling
data_x_test_scal = input_scaling_scal * data_x_test + input_scaling_bias
# Sort test data (only for plotting)
data_x_test_sorted_scal = np.sort(data_x_test_scal,axis=0)

################################################################
# Create and fit svm                                           #
################################################################
# Create NuSVR and set parameters
svm = NuSVR(C=0.1,nu=0.3,kernel='rbf',gamma='auto')
# Fit svm model
svm.fit(data_x_train_scal, data_y_train)

################################################################
# Evaluate model                                               #
################################################################
# Make some test predictions
predictions_test = svm.predict(data_x_test_sorted_scal)

# Plotting
# Plot train data
plt.scatter(data_x_train_scal, data_y_train,c='r')
# Plot test data (sampled from sine -> expected result)
line_sine = plt.plot(data_x_test_sorted_scal, np.sin(data_x_test_sorted_scal), '-b')
plt.setp(line_sine,'linewidth',5)
# Plot predicitons
line_pred = plt.plot(data_x_test_sorted_scal, predictions_test,'-g')
plt.setp(line_pred,'linewidth',5)

plt.xlabel('Input (Scaled)')
plt.ylabel('Target/Output')
plt.legend(['Sine', 'Output Model','Train Data'], loc='upper right')

################################################################
# Export                                                       #
################################################################
# Generate xml string

# The model was trained with scaled input data.
# In order to use the model with non-scaled inputs further parameters (input_scaling_bias, input_scaling_scal) have to be exported.
# One should mind the order of these parameters!
# input_scaling_bias and input_scaling_scal must satisfy the following equation:
# scaled_input_data = input_scaling_scal * non_scaled_input_data + input_scaling_bias

# If no input scaling is used the parameters can be ignored because they are optional.

xml = svm2xml(svm,input_scaling_bias,input_scaling_scal)
# Save string in *.xml-file
with open('ScikitLearnSVMExample.xml','w') as file:
    file.write(xml)

plt.show()