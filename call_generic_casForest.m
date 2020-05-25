
function  call_generic_casForest()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - call_generic_casForest
% Creation Date - 25th May 2020
% Author - Charles Belanger Nzakimuena
% Website - https://www.ibis-space.com/
%
% Description - Function to load data and call generic cascade Deep Forest 
%   functions
%
% Parameters -
%	Input
%
%	Output
%               Cascade deep forest model 
%               Confusion matrix for each fold of K-fold cross-validation
%               K-fold cross-validation performance metrics as excel file
%
% Example -
%		call_generic_casForest()
%
% Acknowledgements -
%           Dedicated to my brother Olivier Belanger-Nzakimuena.
%
% License - BSD
%
% Change History -
%                   25th May 2020 - Creation by Charles Belanger Nzakimuena
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% *add folders containing the RF and ET libraries used*

addpath(genpath('./subFunctions'))

%% *cascade deep forest model files path*

currentFolder = pwd;
% RF and ET models will be stored within the folder at the location
% specified by the path below
mainFolder = fullfile(currentFolder, 'subFunctions');

% cross-validation model performance results are exported to the folder
% at the location specified by the path below
if ~exist(fullfile(currentFolder, 'resultsFolder'), 'dir')
    mkdir(fullfile(currentFolder, 'resultsFolder'));
end
resultsFolder = fullfile(currentFolder, 'resultsFolder');

%% *input data*

load fisheriris

% 'X' represents the numeric matrix of training data, with features for each 
% observation in the form : 
% [(petal length) (petal width) (sepal length) (sepal width)]
X = meas;

% 'Y' represents the class corresponding to each observation
Y = species;

%% *model training*

% convert Y into numerical array (required as input to extra trees model) 
strY = string(Y);
[~,~,numY] = unique(strY);
cellY = cellstr(num2str(numY));

% hyperparameters
hyperP.numLayers = 3; % model number of layers
% RF hyperparameters
hyperP.numTrees_RF = 60; % default 60
% ET hyperparameters
hyperP.numTrees_ET = 60;
hyperP.K_ET = 2; % # of attributes randomly selected at each node; must be <= size(X,2)
hyperP.nmin_ET = 2; % minimum sample size for splitting a node

% Create indices for the k-fold cross-validation
numFolds = 3;
indices = crossvalind('Kfold',Y,numFolds);

% get table row count
rowCount = numFolds+1;
col = zeros(rowCount,1);
colc = cell(rowCount,1);
resultsTable = table(colc,col,col,col,...
    'VariableNames',{'fold' 'recall' 'precision' ...
    'accuracy'});
resultsProfile = zeros(numFolds,3);
tableRow = 0;

precision = zeros(1,max(numY));
recall = zeros(1,max(numY));
for i = 1:numFolds
    test = (indices == i);
    train = ~test;
    
    % *per B-scan performance results*
    deepCas_gen(mainFolder, X(train,:), numY(train,:), hyperP)
    class = deepCas_pred(mainFolder, X(test,:), numY);
    
    % the confusion matrix is generated here for each given 'fold' by
    % comparing 'class' with 'species(test,:)' at each iteration
    confMat = confusionmat(cellY(test,:),class);% ,'order',{'1','2'});
    figure; plotConfMat(confMat.');

    subAccuracy = trace(confMat)/sum(confMat(:));
    for ii = 1:size(confMat,1)
        precision(ii)=confMat(ii,ii)/sum(confMat(ii,:));
    end
    precision(isnan(precision))=[];
    subPrecision = sum(precision)/size(confMat,1);
    for iii = 1:size(confMat,1)
        recall(iii)=confMat(iii,iii)/sum(confMat(:,iii));
    end
    recall(isnan(recall))=[];
    subRecall = sum(recall)/size(confMat,1);

    resultsProfile(i,:) = [subRecall subPrecision subAccuracy];
    tableRow = tableRow + 1;
    resultsTable{tableRow,'fold'} = {num2str(i)};
    resultsTable{tableRow,'recall'} = subRecall;
    resultsTable{tableRow,'precision'} = subPrecision;
    resultsTable{tableRow,'accuracy'} = subAccuracy;
   
end

resultsTable{rowCount,'fold'} = {'average'};
resultsTable{rowCount,'recall'} = mean(resultsProfile(:,1));
resultsTable{rowCount,'precision'} = mean(resultsProfile(:,2));
resultsTable{rowCount,'accuracy'} = mean(resultsProfile(:,3));

fileName1 = fullfile(resultsFolder,'resultsTable.xls');
writetable(resultsTable,fileName1)
