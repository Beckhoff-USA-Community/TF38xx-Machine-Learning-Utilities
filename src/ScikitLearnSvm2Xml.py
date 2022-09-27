#############################################################################################
# TwinCAT 3 Machine Learning: ScikitLearnSvm2Xml.py				   							#
#															  								#
# xml_string = svm2xml(svm, input_scaling_bias, input_scaling_scal) exports				    #
# a scikit-learn svm to a xml-string.													    #
#																							#
# Specify svm as a scikit-learn SVC, NuSVC, OneClassSVM, SVR or NuSVR. For more detailed 	#
# information about supported model characteristics please refer to the documentation.		#
#                                                                                           #
# Specify input_scaling_bias and input_scaling_scal as lists, int or float.				    #
# In order to use the trained model with non-scaled inputs these parameters have to be      #
# exported.                                                                                 #
# Note: 																					#
# - The number of elements of each of these parameters must match the input dimension of    #
#   the model.																	            #
# - Mind the order of these parameters.														#
# - The parameters must satisfy the following equation:										#
#	scaled_input_data = input_scaling_scal * non_scaled_input_data + input_scaling_bias     #
# - input_scaling_bias and input_scaling_scal are optional and can be ignored if no 		#
# 	input scaling is used.																	#
#                                                                                           #
# Exporter tested with the following versions:                                              #
# Python 3.7.6 scikit-learn 0.22.0                                                          #
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

from warnings import warn
import numpy as np
from sklearn.svm import SVC, NuSVC, OneClassSVM, SVR, NuSVR

#############################################################################################
# Functions                                                   	 							#
#############################################################################################

def svm2xml(svm, input_scaling_bias=None, input_scaling_scal=None):

    # check if the following parameters are valid: input scaling, rho, dual coefficients, decision_function_shape
    func_CheckParams(svm, input_scaling_bias, input_scaling_scal)

    xml = ''
    xml = xml + '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml = xml + '<MLlib_XML_File>\n'
    xml = xml + '\t<MachineLearningModel modelName="Support_Vector_Machine" defaultEngine="svm_fp64_engine">\n'

    # auxiliary specifications - product version & target version information, input scaling (optional)
    xml = xml + '\t\t' + '<AuxiliarySpecifications>\n'
    xml = xml  + '\t\t\t' + '<PTI str_producer="Beckhoff MLlib Scikit-learn Exporter" str_producerVersion="' + __producer_version__ + '"/>\n'

    if input_scaling_bias is not None and input_scaling_scal is not None:
        xml = xml + func_BiasScal2XmlIOModification('\t\t', input_scaling_bias, input_scaling_scal)

    xml = xml + '\t\t' + '</AuxiliarySpecifications>\n'

    # model configuration - kernel config, number of input attributes ...
    xml = xml + func_Svm2XmlConf(svm, '\t\t')
    xml = xml + '\t\t<Parameters str_engine="svm_fp64_engine">\n'
    xml = xml + func_Svm2XmlModelInfo(svm, '\t\t\t')

    # probability info - rho, ...
    xml_temp, dual_coef = func_Svm2XmlProbInfo(svm,'\t\t\t')
    xml = xml + xml_temp

    # support vectors
    xml = xml + '\t\t\t<SupportVectors str_type="fp64" int64_rows="' + str(np.shape(svm.support_vectors_)[0]) + '" int64_columns="' + str(np.shape(svm.support_vectors_)[1]) + '">\n'
    xml = xml + func_Svm2XmlVectors(svm, '\t\t\t\t')
    xml = xml + '\t\t\t</SupportVectors>\n'

    # dual coefficients
    xml = xml + '\t\t\t<SupportVectorClassCoefficients str_type="fp64" int64_rows="' + str(np.shape(svm.dual_coef_)[0]) + '" int64_columns="' + str(np.shape(svm.dual_coef_)[1]) + '">\n'
    xml = xml + func_DualCoef2XmlCoef(dual_coef, '\t\t\t\t')
    xml = xml + '\t\t\t</SupportVectorClassCoefficients>\n'

    xml = xml + '\t\t</Parameters>\n'
    xml = xml + '\t</MachineLearningModel>\n'
    xml = xml + '</MLlib_XML_File>\n'

    return xml

def func_Svm2XmlProbInfo(svm,tab):
    if np.shape(svm.intercept_)[0] == 1:
        str_arrfp = 'fp64'
    else:
        str_arrfp = 'arrfp64'

    if type(svm) is OneClassSVM or type(svm) is SVR or type(svm) is NuSVR:
        xml = tab + '<ProbabilityInfo ' + str_arrfp + '_rhoConstants="' + str(-svm.intercept_[0]) + '" bool_hasPairwiseProbabilities="false"/>\n'
        dual_coef = svm.dual_coef_
    elif (type(svm) is SVC or type(svm) is NuSVC) and len(svm.classes_) == 2:
        xml = tab + '<ProbabilityInfo ' + str_arrfp + '_rhoConstants="' + func_Row2Str(svm.intercept_) + '" bool_hasPairwiseProbabilities="false"/>\n'
        dual_coef = (-1)*svm.dual_coef_
    elif (type(svm) is NuSVC or type(svm) is SVC) and len(svm.classes_) > 2:
        xml = tab + '<ProbabilityInfo ' + str_arrfp + '_rhoConstants="' + func_Row2Str(-svm.intercept_) + '" bool_hasPairwiseProbabilities="false"/>\n'
        dual_coef = svm.dual_coef_

    return xml, dual_coef

def func_CheckParams(svm,input_scaling_bias,input_scaling_scal):
    types = [float, int, np.int8, np.int32, np.int64, np.float16, np.float32, np.float64]

    if (input_scaling_bias is not None and input_scaling_scal is None) or (input_scaling_bias is None and input_scaling_scal is not None):
        raise Exception('Bias or scaling is None.')
    elif input_scaling_bias is not None and input_scaling_scal is not None:
        if type(input_scaling_bias) == list:
            if type(input_scaling_scal) != list:
                raise Exception('Scaling does not have the expected type list.')
            else:
                for elem in input_scaling_bias:
                    if type(elem) not in types:
                        raise Exception('Elements of bias do not have the expected type float/int.')

                for elem in input_scaling_scal:
                    if type(elem) not in types:
                        raise Exception('Elements of scal do not have the expected type float/int.')

            if len(input_scaling_bias) != len(input_scaling_scal):
                raise Exception('Bias and scaling do not have the same length.')
            elif len(input_scaling_bias) != np.shape(svm.support_vectors_)[1] or len(input_scaling_scal) != np.shape(svm.support_vectors_)[1]:
                raise Exception('Number of elements in bias/scaling does not match input dimension of the svm.')

        elif type(input_scaling_bias) != list:
            if type(input_scaling_bias) not in types or type(input_scaling_scal) not in types:
                raise Exception('Bias/scal does not have the expected type float/int.')

            elif np.shape(svm.support_vectors_)[1] != 1 or np.shape(svm.support_vectors_)[1] != 1:
                raise Exception('Number of elements in bias/scal does not match input dimension of the svm.')

    if 'decision_function_shape' in svm.get_params().keys() and svm.get_params()['decision_function_shape'] == 'ovr':
        raise Exception('decision_function_shape "ovr" not supported. Please use "ovo" instead.')

    if np.isinf(svm.dual_coef_).any() or np.isnan(svm.dual_coef_).any():
        raise Exception('Invalid dual coefficients inf/-inf/nan.')
    elif np.isinf(svm.intercept_).any() or np.isnan(svm.intercept_).any():
        raise Exception('Invalid intercept inf/-inf/nan.')

    return True

def func_BiasScal2XmlIOModification(tab, input_scaling_bias, input_scaling_scal):
    #   <IOModification>
	# 	    <InputTransformation str_type = "SCALED_OFFSET" arrfp64_offsets = "2,3" arrfp64_scalings = "1.6,2"/>
	#   </IOModification>

    xml = ''

    xml = xml + tab + '\t' + '<IOModification>\n'
    if type(input_scaling_bias) == list:
        xml = xml + tab + '\t\t' + '<InputTransformation str_type="SCALED_OFFSET" arrfp64_offsets="' + func_Row2Str(input_scaling_bias) + '" arrfp64_scalings="' + func_Row2Str(input_scaling_scal) + '"/>\n'
    else:
        xml = xml + tab + '\t\t' + '<InputTransformation str_type="SCALED_OFFSET" fp64_offsets="' + str(input_scaling_bias) + '" fp64_scalings="' + str(input_scaling_scal) + '"/>\n'

    xml = xml + tab + '\t' + '</IOModification>\n'

    return xml

def func_Svm2XmlConf(svm, tab):

    # <Configuration str_operationType="SVM_TYPE_ONE_CLASS" fp64_nu="0.5" str_kernelFunction="KERNEL_FN_RBF" fp64_gamma="25" int64_numInputAttributes="3"/>

    xml = tab + '<Configuration ' + func_Svm2XmlOpModeConfig(svm) + ' ' + func_Svm2XmlKernelConfig(svm) + ' int64_numInputAttributes="' + str(np.shape(svm.support_vectors_)[1]) + '"/>\n'
    return xml

def func_Svm2XmlVectors(svm, tab):

    xml = ''
    for idx in range(0,np.shape(svm.support_vectors_)[0]):
        if np.shape(svm.support_vectors_)[1] > 1:
            xml = xml + tab + '<Row' + str(idx+1) + ' arrfp64_vals="' + func_Row2Str(svm.support_vectors_[idx]) + '"/>\n'
        else:
            xml = xml + tab + '<Row' + str(idx + 1) + ' fp64_vals="' + func_Row2Str(svm.support_vectors_[idx]) + '"/>\n'

    return xml

def func_Row2Str(row):
    xml = ''
    xml = xml + ('%16.12e'%(row[0]))
    for idx in range(1,len(row)):
        xml = xml + (',%16.12e'%(row[idx]))
    return xml

def func_Indices2Str(indices):

    xml = str(indices[0])
    for idx in range(1,len(indices)):
        xml = xml + ',' + str(indices[idx])

    return xml

def func_DualCoef2XmlCoef(dual_coef, tab):

    xml = ''
    for idx in range(0,np.shape(dual_coef)[0]):
        if np.shape(dual_coef)[1] > 1:
            xml = xml + tab + '<Row' + str(idx+1) + ' arrfp64_vals="' + func_Row2Str(dual_coef[idx]) + '"/>\n'
        else:
            xml = xml + tab + '<Row' + str(idx + 1) + ' fp64_vals="' + func_Row2Str(dual_coef[idx]) + '"/>\n'

    return xml

def func_Svm2XmlModelInfo(svm, tab):

    # <ModelInfo int64_numClasses="4" int64_numSVs="5" arrint32_classLabels="1,2,3,4" arrint32_numSVsPerClass="1,1,2,1" arrint32_svIndices="3,5,7,8,9" />

    if type(svm) is SVC or type(svm) is NuSVC:

        for label in svm.classes_:
            if type(label) != int and type(label) != np.int16 and type(label) != np.int32 and type(label) != np.int64:
                raise Exception('Class label type has to be integer.')

        xml = tab + '<ModelInfo int64_numClasses="' + str(len(svm.n_support_)) + '" int64_numSVs="' + str(np.shape(svm.support_vectors_)[0]) + '" arrint32_classLabels="' + func_Indices2Str(svm.classes_) + '" arrint32_numSVsPerClass="' + func_Indices2Str(svm.n_support_) + '"/>\n'
    elif type(svm) is OneClassSVM:
        xml = tab + '<ModelInfo int64_numClasses="2" int64_numSVs="' + str(np.shape(svm.support_vectors_)[0]) + '"/>\n'
    else:
        xml = tab + '<ModelInfo int64_numClasses="2" int64_numSVs="' + str(np.shape(svm.support_vectors_)[0]) + '" arrint32_classLabels="1,2" arrint32_numSVsPerClass="1,' + str(np.shape(svm.support_vectors_)[0]-1) + '"/>\n'
    return xml

def func_Svm2XmlKernelConfig(svm):
    params = svm.get_params()

    if params['kernel'] != 'linear':
        if params['gamma'] == 'scale':
            raise Exception('gamma="scale" is not supported')
        elif params['gamma'] == 'auto_deprecated':
            params['gamma'] = 0.0
            warn("Parameter gamma='auto_deprecated'. Note that gamma=0.0 is exported.")
        elif params['gamma'] == 'auto':
            params['gamma'] = 1/(np.shape(svm.support_vectors_)[1])
            warn("Parameter gamma='auto'. Note that gamma=1/n_features is exported - see docs sklearn.")

    if params['kernel'] == 'linear':
        xml = 'str_kernelFunction="KERNEL_FN_LINEAR"'
    elif params['kernel'] == 'rbf':
        xml = 'str_kernelFunction="KERNEL_FN_RBF" fp64_gamma="' + str(params['gamma']) + '"'
    elif params['kernel'] == 'poly':
        xml = 'str_kernelFunction="KERNEL_FN_POLYNOMIAL" fp64_gamma="' + str(params['gamma']) + '" fp64_coef0="' + str(params['coef0']) + '" int64_degree="' + str(params['degree']) + '"'
    elif params['kernel'] == 'sigmoid':
        xml = 'str_kernelFunction="KERNEL_FN_SIGMOID" fp64_gamma="' + str(params['gamma']) + '" fp64_coef0="' + str(params['coef0']) + '"'
    else: # custom kernel or kernel='precomputed' (pass Gram matrix)
        raise Exception('Kernel not supported')
    return xml

def func_Svm2XmlOpModeConfig(svm):
    params = svm.get_params()

    warn('class_weight and sample_weight are not exported')

    if (type(svm) is SVC or type(svm) is SVR or type(svm) is NuSVR) and np.isinf(params['C']):
        raise Exception('C=inf/C=-inf is not supported.')
    elif type(svm) is OneClassSVM:
        xml = 'str_operationType="SVM_TYPE_ONE_CLASS" fp64_nu="' + str(params['nu']) + '"'
    elif type(svm) is SVC:
        xml = 'str_operationType="SVM_TYPE_C_CLASSIFIER" fp64_cost="' + str(params['C']) + '"'
    elif type(svm) is NuSVC:
        xml = 'str_operationType="SVM_TYPE_NU_CLASSIFIER" fp64_nu="' + str(params['nu']) + '"'
    elif type(svm) is SVR:
        xml = 'str_operationType="SVM_TYPE_EPSILON_REGRESSION" fp64_cost="' + str(params['C']) + '" fp64_epsilon="' + str(params['epsilon']) + '"'
    elif type(svm) is NuSVR:
        xml = 'str_operationType="SVM_TYPE_NU_REGRESSION" fp64_cost="' + str(params['C']) + '" fp64_nu="' + str(params['nu']) + '"'
    else:
        raise Exception('OpMode not supported. Please use SVC/SVR with parameter kernel="linear" instead of LinearSVC/LinearSVR')
    return xml