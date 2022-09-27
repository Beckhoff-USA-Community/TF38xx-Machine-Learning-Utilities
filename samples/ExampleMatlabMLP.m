%% Example - Train And Export A Matlab MLP
% 
%  Train and test data are sampled from               
%  sin(x)+gaussian noise so after training the network        
%  should have learned to predict sin(x), where -pi<=x<=pi.                                     

%% Set Filenames And Paths
addpath('../');
Filename = 'MatlabMLPExample.xml';

%% Set Parameter
NumSamplesTrain = 3000; % number of train samples
NumSamplesTest = 1000; % number of test samples
SampleMin = -pi; % min(samples)
SampleMax = pi; % max(samples)
NoiseFactor = 0.1;

NumNeurons = 3;
NumEpochs = 10000;

%% Generate Train And Test Data
DataXTrain = (SampleMax-SampleMin)*rand(1,NumSamplesTrain)+SampleMin; % train inputs
DataXTrainSorted = sort(DataXTrain,'ascend'); % sorted train inputs -> only for plotting
DataYTrain = sin(DataXTrain)+NoiseFactor*randn(1,NumSamplesTrain); % train targets with noise

% Output scaling, e.g normalization
DataYTrainMean = mean(DataYTrain);
DataYTrainStd = std(DataYTrain);
OutputScalingBias = DataYTrainMean;
OutputScalingScal = DataYTrainStd;
DataYTrainScal = (1/OutputScalingScal) * (DataYTrain - OutputScalingBias); % scaled train outputs

DataXTest = (SampleMax-SampleMin)*rand(1,NumSamplesTest)+SampleMin; % test inputs
DataXTestSorted = sort(DataXTest,'ascend'); % sorted test inputs -> only for plotting
DataYTest = sin(DataXTest)+NoiseFactor*randn(1,NumSamplesTest); % test targets with noise
DataYTestScal = (1/OutputScalingScal) * (DataYTest - OutputScalingBias); % scaled test outputs

%% Initialize Model And Set Train Parameters
net = fitnet(NumNeurons);
net.trainFcn = 'trainscg';
net.trainParam.epochs = NumEpochs;
net.trainParam.max_fail = 250;

%% Training
net = train(net,DataXTrain,DataYTrainScal,'useParallel','no','useGPU','no','showResources','yes');

%% Export Trained Model 
% The model was trained with scaled output data.
% In order to automatically undo the scaling of predictions further parameters (OutputScalingBias, OutputScalingScal) have to be exported.
% One should mind the order of these parameters!
% OutputScalingBias and OutputScalingScal must satisfy the following equation:
% NonScaledPredictions = OutputScalingScal * ScaledPredictions + OutputScalingBias

% If no output scaling is used the parameters can be ignored because they are optional.

MatlabMlp2Xml(net,Filename,OutputScalingBias,OutputScalingScal);

%% Plotting
figure;
    subplot(121);
    title('Model Output - Train Data (Scaled)')
    hold on;
    plot(DataXTrain,DataYTrainScal,'.k');
    plot(DataXTrainSorted,(1/OutputScalingScal) * (sin(DataXTrainSorted) - OutputScalingBias),'-b','LineWidth',2);
    plot(DataXTrainSorted,net(DataXTrainSorted),'-r','LineWidth',2);
    xlabel('Input'); ylabel('Target/Output');
    legend('Train Data','Sine','Output Model');
    hold off;
    
    subplot(122);
    title('Model Output - Test Data (Scaled)')
    hold on;
    plot(DataXTest,DataYTestScal,'.k');
    plot(DataXTestSorted,(1/OutputScalingScal) * (sin(DataXTestSorted) - OutputScalingBias),'-b','LineWidth',2);
    plot(DataXTestSorted,net(DataXTestSorted),'-r','LineWidth',2);
    xlabel('Input'); ylabel('Target/Output');
    legend('Test Data','Sine','Output Model');
    hold off;