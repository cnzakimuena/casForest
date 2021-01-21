function [BaggedEnsemble, bagsData, validLabels] = generic_random_forests(X,Y,iNumBags,str_method)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - generic_random_forests
% Creation Date - 6th July 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
% Description - Function to use random forests
%
% Parameters - 
%	Input	
%		X - matrix
%		Y - matrix of response
%		iNumBags - number of bags to use for boostrapping
%		str_method - 'classification' or 'regression'
%
%	Output
%               BaggedEnsemble - ensemble of random forests
%               Plots of out of bag error
%
% Example -
%
%	 load fisheriris
% 	 X = meas;
%	 Y = species;
%	 BaggedEnsemble = generic_random_forests(X,Y,60,'classification')
%	 predict(BaggedEnsemble,[5 3 5 1.8])
%
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
% License - BSD
%
% Change History - 
%                   7th July 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% When 'OOBPredict' is 'On', info on what observations are out of bag for 
% each tree is stored. 


% ***MODIFY HERE DEPENDING ON NUMBER OF FEATURES IN OBSERVATION ARRAY***
% BaggedEnsemble = TreeBagger(iNumBags,X,Y,'OOBPred','On','Method',str_method,'PredictorSelection','curvature','OOBPredictorImportance','on','PredictorNames',{'LW','delta','theta','alpha','beta'});
BaggedEnsemble = TreeBagger(iNumBags,X,Y,'OOBPred','On','Method',str_method,'PredictorSelection','curvature','OOBPredictorImportance','on');


% % UNCOMMENT FOR : plot out of bag prediction error
% oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
% figID = figure;
% plot(oobErrorBaggedEnsemble)
% xlabel 'Number of grown trees';
% ylabel 'Out-of-bag classification error';
% print(figID, '-dpdf', sprintf('randomforest_errorplot_%s.pdf', date));

% Info from turning 'OOBPredict' 'On' as part of the 'TreeBagger' fonction 
% is used by 'oobPrediction' to compute the predicted class probabilities 
% for each tree in the ensemble.
validLabels = oobPredict(BaggedEnsemble);

a = BaggedEnsemble.Trees;
bagsData = getParameters(a, iNumBags);

% % % UNCOMMENT TO HIDE RULES DECISION TREE DIAGRAM : note that there are 'n' number of trees corresponding to iNumBags, where
% % % 'n' is the input to the function 'view(BaggedEnsemble.Trees{n})'
% % % to view the fist tree (n = 1) 
% % view(BaggedEnsemble.Trees{1}) % text description
% view(BaggedEnsemble.Trees{1},'mode','graph') % graphic description
