%  TwinCAT 3 Machine Learning: MatlabMlp2Xml.m	
% 															  								
%  xml_string = MatlabMlp2Xml(net, fnstr, varargin) exports				
%  a Matlab MLP (shallow net API) to a xml-string.		
%
%  Specify net as a feedforward Matlab MLP. For more detailed information about 				
%  supported model characteristics please refer to the documentation.	
%
%  Specify fnstr as output_path + output_filename + .xml .
%
%  If the model is trained with scaled output data use varargin to pass
%  output_scaling_bias and output_scaling_scal (in this order!) in order to
%  automatically undo the scaling.
%  Specify output_scaling_bias and output_scaling_scal as lists, int or float.													
%  Note: 																					
%  - The number of elements of each of these parameters must match the output				
%  	 dimension of the model.																	
%  - Mind the order of these parameters.														
%  - The parameters must satisfy the following equation:
%	 non_scaled_predictions = output_scaling_scal * scaled_predictions + output_scaling_bias	
%  - output_scaling_bias and output_scaling_scal are optional and can be ignored if no 		
%  	 output scaling is used.			
%
%  Tested with:
%  - Matlab Release 2019b (Update 5) Version 9.7, Deep Learning Toolbox Version 13.0
%  - Matlab Release 2020a (Update 2) Version 9.8, Deep Learning Toolbox Version 14.0
% 																							
%  MIT License																				
% 																							
%  Copyright (c) 2020 Beckhoff Automation GmbH & Co. KG 										
% 																							
%  Permission is hereby granted, free of charge, to any person obtaining a					
%  copy of this software and associated documentation files (the "Software"),				
%  to deal in the Software without restriction, including without limitation					
%  the rights to use, copy, modify, merge, publish, distribute, sublicense,					
%  and/or sell copies of the Software, and to permit persons to whom the						
%  Software is furnished to do so, subject to the following conditions:						
% 																							
%  The above copyright notice and this permission notice shall be included in				
%  all copies or substantial portions of the Software.										
% 																							
%  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR				
%  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,					
%  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE				
%  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER					
%  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING					
%  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER						
%  DEALINGS IN THE SOFTWARE.																	
 
%% functions   
function xml = MatlabMlp2Xml(net, fnstr, varargin)
    producerVersion = '3.1.200902.0'; 
    requiredVersion = '3.1.200902.0';
    
    xml = [];
    xml = [xml '<?xml version="1.0" encoding="UTF-8"?>\n'];
    xml = [xml '<MLlib_XML_File>\n'];
    xml = [xml '\t<MachineLearningModel modelName="Mlp_Neural_Network">\n'];
    
    % auxiliary specifications - product version & target version information, input scaling (optional)
    xml = [xml '\t\t' '<AuxiliarySpecifications>\n'];
    xml = [xml '\t\t\t' '<PTI str_producer="Beckhoff MLlib Matlab Exporter" str_producerVersion="' producerVersion '"/>\n'];

    if nargin==4
        bias = varargin{1}; scal = varargin{2};
        if func_TestBiasScal(net,bias,scal) == 1
            xml = [xml func_BiasScal2XmlIOModification('\t\t', bias, scal)];
        end
    elseif nargin>2 && nargin~=4
        error('MLPExporter:UnexcpectedNumberOfInputs','Wrong number of inputs. 2 or 4 inputs are expected.');
    end
    
    xml = [xml '\t\t' '</AuxiliarySpecifications>\n'];
    
    % test input processFcn and output processFcn
    if length(net.inputs{1}.processFcns)>1
        error('MLPExporter:UnexcpectedNumberOfInpProcessFcns','More than one input processFcn. Please use not more than 1.');
    
    elseif length(net.outputs{net.numLayers}.processFcns)>1
        error('MLPExporter:UnexcpectedNumberOfOutProcessFcns','More than one output processFcn. Please use not more than 1.');
    
    elseif ~isempty(net.outputs{net.numLayers}.processFcns) && strcmp(net.layers{net.numLayers}.transferFcn,'purelin') == 0
        error('MLPExporter:MatlabOutScalAndProcessFcnNotSupported','Output processFcn and transferFcn ~= purelin used.');
    
    elseif (~isempty(net.inputs{1}.processFcns) &&(strcmp(net.inputs{1}.processFcns{1},'removeconstantrows')==1 || strcmp(net.inputs{1}.processFcns{1},'removerows')==1 || ...
            strcmp(net.inputs{1}.processFcns{1},'fixunknowns')==1 || strcmp(net.inputs{1}.processFcns{1},'processpca')==1) )
        error('MLPExporter:MatlabInpProcessFcnNotSupported','Input processFcn not supported.');
    
    elseif (~isempty(net.outputs{net.numLayers}.processFcns) &&(strcmp(net.outputs{net.numLayers}.processFcns{1},'removeconstantrows')==1 || strcmp(net.outputs{net.numLayers}.processFcns{1},'removerows')==1 || ...
            strcmp(net.outputs{net.numLayers}.processFcns{1},'fixunknowns')==1 || strcmp(net.outputs{net.numLayers}.processFcns{1},'processpca')==1)) 
        error('MLPExporter:MatlabOutProcessFcnNotSupported','Output processFcn not supported.');
    end
    
    % model configuration - number of layers, uses bias, number of neurons, activation function, ...
    xml = [xml func_Net2XmlConf(net, '\t\t')];
    xml = [xml '\t\t<Parameters str_engine="mlp_fp32_engine" int_numLayers="' num2str(net.numLayers) '" bool_usesBias="' func_net2strusesBias(net) '">\n'];
    
    % layer weights and additional layer information
    for i = 1:net.numLayers
        xml = [xml func_Net2XmlLayer(net, i, '\t\t\t')];
    end
    
    xml = [xml '\t\t</Parameters>\n'];
    xml = [xml '\t</MachineLearningModel>\n'];
    xml = [xml '</MLlib_XML_File>\n'];
    
    % create *.xml-file
    if ~isempty(fnstr)
        fileID = fopen(fnstr,'w');
        fprintf(fileID,xml);
        fclose(fileID);
    else
        warning('MLPExporter:OuputPathEmpty','Output path empty -> xml string not saved');
    end
end

function boolBiasScalOk = func_TestBiasScal(net,bias,scal)
    boolBiasScalOk = 0;
    biasSize = size(bias); scalSize = size(scal);
        
    if strcmp(class(bias),'double') == 0 || strcmp(class(scal),'double') == 0
        error('MLPExporter:BiasScalUnexpectedClass','Bias/scal does not have the expected class double.');
    elseif biasSize(1) ~= 1 || scalSize(1) ~= 1
        error('MLPExporter:BiasScalUnexpectedSize','Bias/scal does not have the expected size [1 *].');
    elseif biasSize(2) ~= scalSize(2)
        error('MLPExporter:BiasScalSizeNotEqual','Bias and scal do not have the same size.');
    elseif biasSize(2) ~= net.layers{net.numLayers}.size
        error('MLPExporter:BiasScalSizeNotMatchingOutputDimNet','Number of elements in bias/scal does not match output dimension of the network.');
    else
        boolBiasScalOk = 1;
    end
end

function xml = func_BiasScal2XmlIOModification(tab, bias, scal)
    %   <IOModification>
    %       <OutputTransformation str_type = "SCALED_OFFSET" arrfp64_offsets = "2,3" arrfp64_scalings = "1.6,2"/>
    %   </IOModification>
    
    if length(bias)>1
        strFp64Vals = 'arrfp64';
    else
        strFp64Vals = 'fp64';
    end

    xml = [];
    xml = [xml tab '\t' '<IOModification>\n'];
    xml = [xml tab '\t\t' '<OutputTransformation str_type="SCALED_OFFSET" ' strFp64Vals '_offsets="' func_Row2Str(bias) '" ' strFp64Vals '_scalings="' func_Row2Str(scal) '"/>\n'];    
    xml = [xml tab '\t' '</IOModification>\n'];
end

function xml = func_Net2XmlConf(net, tab)

    % <Configuration Configuration="4" int_numLayers="4" bool_usesBias="true">
    %     <MlpLayer1 int_numNeurons="6" str_activationFunction="ACT_FN_SIGMOID"/>
    %     <MlpLayer2 int_numNeurons="6" str_activationFunction="ACT_FN_SIGMOID"/>
    %     <MlpLayer3 int_numNeurons="6" str_activationFunction="ACT_FN_SIGMOID"/>
    %     <MlpLayer4 int_numNeurons="3" str_activationFunction="ACT_FN_SIGMOID"/>
    % </Configuration>

    xml = [];
    xml = [xml tab '<Configuration int_numInputNeurons="' num2str(net.input.size) '" int_numLayers="' num2str(net.numLayers) '" bool_usesBias="' func_net2strusesBias(net) '">\n'];
    for i = 1:net.numLayers
        xml = [xml tab '\t' '<MlpLayer' num2str(i) ' int_numNeurons="' num2str(net.layers{i}.size) '" str_activationFunction="' func_Str2TransFct(net.layers{i}.transferFcn) '"/>\n'];
    end
    xml = [xml tab '</Configuration>\n'];
    
end

function xml = func_Net2XmlLayer(net, layer, tab)

    % <MlpLayer1 str_type="dense" str_activationFunction="ACT_FN_SIGMOID">
    %     <WeightMatrix str_type="fp32" int_rows="6" int_columns="5">
    %         <Row1 arrfp32_vals="0.455443,0.301906,0.541781,-0.253339,-0.639187"/>
    %         <Row2 arrfp32_vals="0.891971,-0.129105,0.357915,-0.744195,-0.277263"/>
    %         <Row3 arrfp32_vals="-0.347602,0.323089,-0.912997,-0.23274,0.314637"/>
    %         <Row4 arrfp32_vals="-0.211747,-0.636116,-0.736092,0.0251508,0.696611"/>
    %         <Row5 arrfp32_vals="-0.099763,-0.42249,-0.127794,0.311451,-0.75072"/>
    %         <Row6 arrfp32_vals="0.692093,0.819697,-0.30946,0.30172,0.330007"/>
    %     </WeightMatrix>
    % </MlpLayer1>

    if layer == 1
        inpMat = net.IW{layer}; [numRows,numCols] = size(inpMat);
        inpB = net.b{layer};
        
        if ~isempty(net.inputs{layer}.processFcns)
            gain = net.inputs{1}.processSettings{1}.gain;
            xoffset = net.inputs{1}.processSettings{1}.xoffset;
            
            % mapminmax
            if strcmp(net.inputs{layer}.processFcns{1},'mapminmax')==1            
                ym = net.inputs{1}.processSettings{1}.ymin;
            % mapstd    
            elseif strcmp(net.inputs{layer}.processFcns{1},'mapstd')==1
                ym = net.inputs{1}.processSettings{1}.ymean;
            end

            Ascal = diag(gain);
            Bscal = (ym-diag(gain)*xoffset);
            
            if ~isempty(inpB)
                inpB = inpMat*Bscal + inpB;
            else
                inpB = inpMat*Bscal;
            end
            inpMat = inpMat*Ascal;

        end
        
    elseif layer == net.numLayers && ~isempty(net.outputs{net.numLayers}.processFcns)
        inpMat = net.LW{layer,layer-1}; [numRows,numCols] = size(inpMat);
        inpB = net.b{layer};

        gain = net.outputs{net.numLayers}.processSettings{1}.gain;
        xoffset = net.outputs{net.numLayers}.processSettings{1}.xoffset;

        % mapminmax
        if strcmp(net.outputs{net.numLayers}.processFcns{1},'mapminmax')==1            
            ym = net.outputs{net.numLayers}.processSettings{1}.ymin;
        % mapstd    
        elseif strcmp(net.outputs{net.numLayers}.processFcns{1},'mapstd')==1
            ym = net.outputs{net.numLayers}.processSettings{1}.ymean;
        end

        Ascal = inv(diag(gain));
        if ~isempty(inpB)
            inpB = Ascal*(inpB-ym)+xoffset;
        else
            inpB = Ascal*(-ym) + xoffset;
        end
        inpMat = Ascal*inpMat;
        
    else
        inpMat = net.LW{layer,layer-1}; [numRows,numCols] = size(inpMat);
        inpB = net.b{layer};
    end
    
    xml = [];
    xml = [xml tab '<MlpLayer' num2str(layer) ' str_type="dense" str_activationFunction="' func_Str2TransFct(net.layers{layer}.transferFcn) '">\n'];
    
    if ~isempty(inpB)
        % every layer has a bias
        xml = [xml tab '\t<WeightMatrix str_type="fp32" int_rows="' num2str(numRows) '" int_columns="' num2str(numCols+1) '">\n'];
        for i = 1:numRows
            xml = [xml tab '\t \t <Row' num2str(i) ' arrfp32_vals="' func_Row2Str(inpMat(i,:)) ',' sprintf('%16.12e',inpB(i)) '"/>\n'];
        end
        
    elseif isempty(inpB) && strcmp(func_net2strusesBias(net),'true')==1
        % If there is at least one layer with a bias, but the
        % current layer does not have a bias add a column with zeros
        xml = [xml tab '\t<WeightMatrix str_type="fp32" int_rows="' num2str(numRows) '" int_columns="' num2str(numCols+1) '">\n'];
        for i = 1:numRows
            xml = [xml tab '\t \t <Row' num2str(i) ' arrfp32_vals="' func_Row2Str(inpMat(i,:)) ',' sprintf('%16.12e',0.0) '"/>\n'];
        end
        
    elseif strcmp(func_net2strusesBias(net),'false')==1
        % no layer has a bias 
        xml = [xml tab '\t<WeightMatrix str_type="fp32" int_rows="' num2str(numRows) '" int_columns="' num2str(numCols) '">\n'];
        
        if length(inpMat(1,:))>1
            str_vals = 'arrfp32_vals';
        else
            str_vals = 'fp32_vals';
        end
        
        for i = 1:numRows
            xml = [xml tab '\t \t <Row' num2str(i) ' ' str_vals '="' func_Row2Str(inpMat(i,:)) '"/>\n'];
        end
    end
    
    xml = [xml tab '\t</WeightMatrix>\n'];
    xml = [xml tab '</MlpLayer' num2str(layer) '>\n']; 
end

function xml = func_Row2Str(row)
    xml = [];
    xml = [xml sprintf('%16.12e',row(1))];
    for i = 2:length(row)
        xml = [xml ',' sprintf('%16.12e',row(i))];
    end
end

function xml = func_Str2TransFct(str)
    switch str
        case 'logsig'
            xml = 'ACT_FN_SIGMOID';
        case 'tansig'
            xml = 'ACT_FN_TANH';
        case 'softmax'
            xml = 'ACT_FN_SOFTMAX';
        case 'poslin'
            xml = 'ACT_FN_RELU';
        case 'purelin'
            xml = 'ACT_FN_IDENTITY';
        otherwise
            error('MLPExporter:TransferFunc','Transfer function not supported');
    end
end

function xml = func_net2strusesBias(net)
    % Return 'true', if at least one layer has a bias or net has input/output scaling, else: return 'false'
    
    if sum(net.biasConnect)>=1 || ~isempty(net.inputs{1}.processFcns) || ~isempty(net.outputs{net.numLayers}.processFcns)
        xml = 'true';
    else
        xml = 'false';
    end
end

