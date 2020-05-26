
function final_pred = deepCas_pred(mainFolder, X, numY)
% predict with deep forest model which combines random forest and extra trees

% get total # of possible classes
classNum = max(numY);

ave_ouput = zeros(size(X,1),classNum);
final_prob = zeros(size(X,1),1); 
final_pred = zeros(size(X,1),1); 

modelFolder = fullfile(mainFolder, 'deepForest');

modelDir = dir(modelFolder);
dirFlags = [modelDir .isdir] & ~strcmp({modelDir.name},'.') & ~strcmp({modelDir .name},'..');
nameFolds = modelDir(dirFlags);
numLayers = numel(nameFolds);

% get total # of possible classes 
classNum = max(numY);

% iteratively load model folders and remove model variables everytime predictions are obtained	
for i = 1:numLayers
    
    layerFolder = fullfile(modelFolder, ['layer' num2str(i,'%.f')]);
    
    % specify prediction input
    if i == 1
        predX = X;
    else
        predX = nextInput;
    end
    
    % current layer random forest prediction
    load(fullfile(layerFolder, 'model_1_1.mat'));
    [~,score_1_1] = predict(model_1_1, predX);
    load(fullfile(layerFolder, 'model_1_2.mat'));
    [~,score_1_2] = predict(model_1_1, predX);
    clear model_1_1 model_1_2
    
    % current layer extra trees prediction
    inputType_ET = logical(ones(size(predX,2),1));
    load(fullfile(layerFolder, 'model_2_1.mat'));
    [~, score_2_1] = predictWithAnEnsemble(model_2_1,predX,1,classNum);
    load(fullfile(layerFolder, 'model_2_2.mat'));
    [~, score_2_2] = predictWithAnEnsemble(model_2_2,predX,1,classNum);
    clear model_2_1 model_2_2
    
    if i ~= numLayers
        nextInput = [X score_1_1 score_1_2 score_2_1 score_2_2];
    elseif i == numLayers
        for q = 1:size(X,1)
            for r = 1:classNum
                ave_ouput(q,r) = mean([score_1_1(q,r) score_1_2(q,r) score_2_1(q,r) score_2_2(q,r)]);
                [final_prob(q), final_pred(q)] = max(ave_ouput(q,:));
            end
        end
    end
    
end

final_pred = cellstr(num2str(final_pred));

end
    
