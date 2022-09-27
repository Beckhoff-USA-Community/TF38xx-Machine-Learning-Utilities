#############################################################################################
# TwinCAT 3 Machine Learning: KerasMlp2Xml.py				   								#
#															  								#
# xml_string = net2xml(net, output_scaling_bias, output_scaling_scal) exports				#
# a Keras sequential model to a xml-string.													#
#																							#
# Specify net as a sequential Keras model. For more detailed information about 				#
# supported model characteristics please refer to the documentation.						#
#																							#
# Specify output_scaling_bias and output_scaling_scal as lists, int or float.				#
# If the model is trained with scaled output data use these parameters to 					#
# automatically undo the scaling of future predictions.										#
# Note: 																					#
# - The number of elements of each of these parameters must match the output				#
# 	dimension of the model.																	#
# - Mind the order of these parameters.														#
# - The parameters must satisfy the following equation:										#
#	non_scaled_predictions = output_scaling_scal * scaled_predictions + output_scaling_bias	#
# - output_scaling_bias and output_scaling_scal are optional and can be ignored if no 		#
# 	output scaling is used.																	#
#                                                                                           #
# Exporter tested with the following versions:                                              #
# - Python: 3.7.5, Keras: 2.2.4                                                             #
#																							#
# MIT License																				#
# 																							#
# Copyright (c) 2020 Beckhoff Automation GmbH & Co. KG 										#
# 																							#
# Permission is hereby granted, free of charge, to any person obtaining a					#
# copy of this software and associated documentation files (the "Software"),				#
# to deal in the Software without restriction, including without limitation					#
# the rights to use, copy, modify, merge, publish, distribute, sublicense,					#
# and/or sell copies of the Software, and to permit persons to whom the						#
# Software is furnished to do so, subject to the following conditions:						#
# 																							#
# The above copyright notice and this permission notice shall be included in				#
# all copies or substantial portions of the Software.										#
# 																							#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR				#
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,					#
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE				#
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER					#
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING					#
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER						#
# DEALINGS IN THE SOFTWARE.																	#
#############################################################################################

__producer_version__ = '3.1.200902.0'
__required_version__ = '3.1.200902.0'

#############################################################################################
# Import                                                       								#
#############################################################################################
import numpy as np
from warnings import warn
import tensorflow.python.keras.engine as engine

#############################################################################################
# Functions                                                   	 							#
#############################################################################################
def net2xml(net,output_scaling_bias=None,output_scaling_scal=None):

	# test model type
	if type(net) != engine.sequential.Sequential:
		raise Exception('Unexpected model type: {}. Expected: tf.python.keras.engine.sequential.Sequential'.format(type(net)))

	# test layer types & get number and indices of Dense layers
	idxDenseLayer, numDenseLayer = func_GetDenseLayerInfos(net)

	xml = ''
	xml = xml + '<?xml version="1.0" encoding="UTF-8"?>\n'
	xml = xml + '<MLlib_XML_File>\n'
	xml = xml + '\t<MachineLearningModel modelName="Mlp_Neural_Network">\n'

	# auxiliary specifications - product version & target version information, output scaling (optional)
	xml = xml + '\t\t' + '<AuxiliarySpecifications>\n'
	xml = xml + '\t\t\t' + '<PTI str_producer="Beckhoff MLlib Keras Exporter" str_producerVersion="' + __producer_version__ + '"/>\n'

	if output_scaling_bias is not None and output_scaling_scal is not None:
		idxLastDenseLayer = idxDenseLayer[-1]
		func_CheckBiasScal(net, output_scaling_bias, output_scaling_scal, idxLastDenseLayer)
		xml = xml + func_BiasScal2XmlIOModification('\t\t', output_scaling_bias, output_scaling_scal)

	xml = xml + '\t\t' + '</AuxiliarySpecifications>\n'

	# model configuration - number of layers, uses bias, number of neurons, activation function, ...
	xml = xml + func_Net2XmlConf(net, '\t\t', numDenseLayer, idxDenseLayer)
	xml = xml + '\t\t<Parameters str_engine="mlp_fp32_engine" int_numLayers="' + str(numDenseLayer) + '" bool_usesBias="' + func_UsesBias2Str(net,idxDenseLayer) + '">\n'

	# layer weights and additional layer information
	counter = 1
	for i in range(0,numDenseLayer):
		xml = xml + func_Net2XmlLayer(net, idxDenseLayer[i], counter, '\t\t\t', idxDenseLayer)
		counter+=1

	xml = xml + '\t\t</Parameters>\n'
	xml = xml + '\t</MachineLearningModel>\n'
	xml = xml + '</MLlib_XML_File>\n'

	return xml

def func_CheckBiasScal(net, output_scaling_bias, output_scaling_scal, idxLastDenseLayer):
	types = [float, int, np.int8, np.int32, np.int64, np.float16, np.float32, np.float64]

	if (output_scaling_bias is not None and output_scaling_scal is None) or (output_scaling_bias is None and output_scaling_scal is not None):
		raise Exception('Bias or scaling is None.')

	elif type(output_scaling_bias) == list:

		if type(output_scaling_scal) != list:
			raise Exception('Scaling does not have the expected type list.')

		elif len(output_scaling_bias) != len(output_scaling_scal):
			raise Exception('Bias and scaling do not have the same length.')

		else:
			for elem in output_scaling_bias:
				if type(elem) not in types:
					raise Exception('Elements of bias do not have the expected type float/int.')

			for elem in output_scaling_scal:
				if type(elem) not in types:
					raise Exception('Elements of scal do not have the expected type float/int.')

		if len(output_scaling_bias) != net.layers[idxLastDenseLayer].get_config()['units'] or len(output_scaling_scal) != \
				net.layers[idxLastDenseLayer].get_config()['units']:
			raise Exception('Number of elements in bias/scaling does not match output dimension of the network.')

	elif type(output_scaling_bias) != list:
		if type(output_scaling_bias) not in types or type(output_scaling_scal) not in types:
			raise Exception('Bias/scal does not have the expected type float/int.')

		elif net.layers[idxLastDenseLayer].get_config()['units'] != 1 or net.layers[idxLastDenseLayer].get_config()['units'] != 1:
			raise Exception('Number of elements in bias/scaling does not match output dimension of the network.')

	return True

def func_GetDenseLayerInfos(net):
	config = net.get_config()
	idxDenseLayer = []
	numDenseLayer = 0
	for idxLayer in range(0, np.shape(net.layers)[0]):
		if config['layers'][idxLayer]['class_name'] == 'Dense':
			idxDenseLayer.append(idxLayer)
			numDenseLayer += 1
		elif config['layers'][idxLayer]['class_name'] == 'Dropout':
			warn('Dropout layer is not exported.')
		else:
			raise Exception('Invalid layer type: {}'.format(config['layers'][idxLayer]['class_name']))

	return idxDenseLayer, numDenseLayer

def func_BiasScal2XmlIOModification(tab, output_scaling_bias, output_scaling_scal):
	# <AuxiliarySpecifications>
	#   <IOModification>
	# 		<OutputTransformation str_type = "SCALED_OFFSET" arrfp64_offsets = "2,3" arrfp64_scalings = "1.6,2"/>
	#   </IOModification>
	# </AuxiliarySpecifications>

	xml = ''
	xml = xml + tab + '\t' + '<IOModification>\n'
	if type(output_scaling_bias) == list:
		xml = xml + tab + '\t\t' + '<OutputTransformation str_type="SCALED_OFFSET" arrfp64_offsets="' + func_Row2Str(output_scaling_bias) + '" arrfp64_scalings="' + func_Row2Str(output_scaling_scal) + '"/>\n'
	else:
		xml = xml + tab + '\t\t' + '<OutputTransformation str_type="SCALED_OFFSET" fp64_offsets="' + str(output_scaling_bias) + '" fp64_scalings="' + str(output_scaling_scal) + '"/>\n'

	xml = xml + tab + '\t' + '</IOModification>\n'

	return xml

def func_Net2XmlConf(net, tab, numDenseLayer, idxDenseLayer):

    # <Configuration Configuration="4" int_numLayers="4" bool_usesBias="true">
    #     <MlpLayer1 int_numNeurons="6" str_activationFunction="ACT_FN_SIGMOID"/>
    #     <MlpLayer2 int_numNeurons="6" str_activationFunction="ACT_FN_SIGMOID"/>
    #     <MlpLayer3 int_numNeurons="6" str_activationFunction="ACT_FN_SIGMOID"/>
    #     <MlpLayer4 int_numNeurons="3" str_activationFunction="ACT_FN_SIGMOID"/>
    # </Configuration>
	
    inputSize = np.shape(net.layers[0].get_weights()[0])[0]
	
    xml = ''
    xml = xml + tab + '<Configuration int_numInputNeurons="' + str(inputSize) + '" int_numLayers="' + str(numDenseLayer) + '" bool_usesBias="' + func_UsesBias2Str(net,idxDenseLayer) + '">\n'

    counter = 1
    for i in idxDenseLayer[0:numDenseLayer]:
        config = net.layers[i].get_config()
        layerSize = config['units']
        layAct = config['activation']
        xml = xml + tab + '\t' + '<MlpLayer' + str(counter) + ' int_numNeurons="' + str(layerSize) + '" str_activationFunction="' + func_Str2TransFct(layAct) + '"/>\n'
        counter+=1
    xml = xml + tab + '</Configuration>\n'

    return xml

def func_Net2XmlLayer(net, layer, counter, tab, idxDenseLayer):

	# <MlpLayer1 str_type="dense" str_activationFunction="ACT_FN_SIGMOID">
    #     <WeightMatrix str_type="fp32" int_rows="6" int_columns="5">
    #         <Row1 arrfp32_vals="0.455443,0.301906,0.541781,-0.253339,-0.639187"/>
    #         <Row2 arrfp32_vals="0.891971,-0.129105,0.357915,-0.744195,-0.277263"/>
    #         <Row3 arrfp32_vals="-0.347602,0.323089,-0.912997,-0.23274,0.314637"/>
    #         <Row4 arrfp32_vals="-0.211747,-0.636116,-0.736092,0.0251508,0.696611"/>
    #         <Row5 arrfp32_vals="-0.099763,-0.42249,-0.127794,0.311451,-0.75072"/>
    #         <Row6 arrfp32_vals="0.692093,0.819697,-0.30946,0.30172,0.330007"/>
    #     </WeightMatrix>
    # </MlpLayer1>
	
	weightsMat = net.layers[layer].get_weights()[0].transpose()
	if len(net.layers[layer].get_weights()) > 1:
		biasMat = net.layers[layer].get_weights()[1]
	else:
		biasMat = None
				
	numRows = np.shape(weightsMat)[0]
	numCols = np.shape(weightsMat)[1]
	
	config = net.layers[layer].get_config()
	layAct = config['activation']
	
	xml = ''
	xml = xml + tab + '<MlpLayer' + str(counter) + ' str_type="dense" str_activationFunction="' + func_Str2TransFct(layAct) + '">\n'

	if biasMat is not None:
		# layer has a bias
		xml = xml + tab + '\t<WeightMatrix str_type="fp32" int_rows="' + str(numRows) + '" int_columns="' + str(numCols + 1) + '">\n'
		for i in range(0, numRows):
			xml = xml + tab + '\t \t <Row' + str(i + 1) + ' arrfp32_vals="' + func_Row2Str(weightsMat[i, :]) + ', %16.12e"/>\n'%((biasMat[i]))

	elif biasMat is None and func_UsesBias2Str(net, idxDenseLayer) == 'true':
		# If there is at least one layer with a bias in the model, but the current layer does not have a bias add a column with
		xml = xml + tab + '\t<WeightMatrix str_type="fp32" int_rows="' + str(numRows) + '" int_columns="' + str(numCols + 1) + '">\n'
		for i in range(0, numRows):
			xml = xml + tab + '\t \t <Row' + str(i + 1) + ' arrfp32_vals="' + func_Row2Str(weightsMat[i, :]) + ', %16.12e"/>\n'%(0.0)
	else:
		# no layer has a bias
		xml = xml + tab + '\t<WeightMatrix str_type="fp32" int_rows="' + str(numRows) + '" int_columns="' + str(numCols) + '">\n'
		for i in range(0, numRows):
			if np.shape(weightsMat)[1]>1:
				xml = xml + tab + '\t \t <Row' + str(i + 1) + ' arrfp32_vals="' + func_Row2Str(weightsMat[i, :]) + '"/>\n'
			else:
				xml = xml + tab + '\t \t <Row' + str(i + 1) + ' fp32_vals="' + func_Row2Str(weightsMat[i, :]) + '"/>\n'

	xml = xml + tab + '\t</WeightMatrix>\n'
	xml = xml + tab + '</MlpLayer' + str(counter) + '>\n'
	
	return xml
	
def func_Row2Str(row):
	
	numColumns = np.shape(row)[0]
	
	xml = ''
	xml = xml + ('%16.12e'%(row[0]))
	for i in range(1,numColumns):
		xml = xml + ', %16.12e'%(row[i])
		
	return xml

def func_UsesBias2Str(net, idxDenseLayer):
	config = net.get_config()

	if sum([config['layers'][idxLayer]['config']['use_bias'] for idxLayer in idxDenseLayer])>=1:
		return 'true'
	else:
		return 'false'

def func_Str2TransFct(str):
	
	if str == 'sigmoid':
		xml = 'ACT_FN_SIGMOID'
	elif str == 'exponential':
		xml = 'ACT_FN_EXP'
	elif str == 'tanh':
		xml = 'ACT_FN_TANH'
	elif str == 'relu':
		xml = 'ACT_FN_RELU'
	elif str == 'softmax':
		xml = 'ACT_FN_SOFTMAX'
	elif str == 'softplus':
		xml = 'ACT_FN_SOFTPLUS'
	elif str == 'softsign':
		xml = 'ACT_FN_SOFTSIGN'
	elif str == 'linear':
		xml = 'ACT_FN_IDENTITY'
	else:
		raise Exception('Not supported activation ' + str + ' used.\n')
	
	return xml