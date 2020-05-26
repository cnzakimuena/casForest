
function deepCas_gen(mainFolder, X, numY, hyperP)
% trains a deep forest model which combines random forest and extra trees

% get total # of possible classes
classNum = max(numY);

problemType_ET = 1;  % classification problem

% create a model folder
if ~exist(fullfile(mainFolder, 'deepForest'), 'dir')
    mkdir(fullfile(mainFolder, 'deepForest'));
end
modelFolder = fullfile(mainFolder, 'deepForest');

for i = 1:hyperP.numLayers
    
    % create a layer folder wihtin the model folder
    if ~exist(fullfile(modelFolder, ['layer' num2str(i,'%.f')]), 'dir')
        mkdir(fullfile(modelFolder, ['layer' num2str(i,'%.f')]));
    end
    layerFolder = fullfile(modelFolder, ['layer' num2str(i,'%.f')]);
    
    % specify training input
    if i == 1
        trainXY = [X numY];
    else
        trainXY = nextInput;
    end
    
    % current layer random forest models generation
    model_1_1 = generic_random_forests(trainXY(:,1:end-1),trainXY(:,end),hyperP.numTrees_RF,'classification');
    save(fullfile(layerFolder, 'model_1_1.mat'),'model_1_1')
    [~,score_1_1] = predict(model_1_1, trainXY(:,1:end-1));
    model_1_2 = generic_random_forests(trainXY(:,1:end-1),trainXY(:,end),hyperP.numTrees_RF,'classification');
    save(fullfile(layerFolder, 'model_1_2.mat'),'model_1_2')
    [~,score_1_2] = predict(model_1_2, trainXY(:,1:end-1));
    
    % current layer extra trees models generation
    inputType_ET = logical(ones(size(trainXY(:,1:end-1),2),1));
    [model_2_1,~,~,~] = buildAnEnsemble(hyperP.numTrees_ET, hyperP.K_ET,hyperP.nmin_ET,trainXY,problemType_ET,inputType_ET,[]);
    [~, score_2_1] = predictWithAnEnsemble(model_2_1,trainXY(:,1:end-1),1,classNum);
    save(fullfile(layerFolder, 'model_2_1.mat'),'model_2_1')
    [model_2_2,~,~,~] = buildAnEnsemble(hyperP.numTrees_ET, hyperP.K_ET,hyperP.nmin_ET,trainXY,problemType_ET,inputType_ET,[]);
    [~, score_2_2] = predictWithAnEnsemble(model_2_2,trainXY(:,1:end-1),1,classNum);
    save(fullfile(layerFolder, 'model_2_2.mat'),'model_2_2')
    
    nextInput = [X score_1_1 score_1_2 score_2_1 score_2_2 numY];
    
end

end
