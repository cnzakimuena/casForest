
function treesStruc = getParameters(treesData, numTrees)

treesStruc = cell(numTrees,4);
for k = 1:numTrees
    
    % to obtain tree levels array
    treeData = treesData{k};
    nodeLevels = zeros(size(treeData.Parent));
    currParent0 = 0;
    currLevel = 1;
    nodeLevels(1) = currLevel-1;
    for i = 2:treeData.NumNodes
        % assign node parent to a variable
        currParent = treeData.Parent(i);
        % determine if current parent is different from previous parent
        if currParent ~= currParent0
            % if current parent is different from last parent, assign current
            % parent, 'currParent' as initial parent, 'currParent0'
            currParent0 = currParent;
            % initialize maximim spots count
            remSpots = 2;
        end
        remSpots = remSpots - 1;
        nodeLevels(i) = currLevel;
        if remSpots == 0 && i ~= treeData.NumNodes
            if nodeLevels(treeData.Parent(i)) ~= nodeLevels(treeData.Parent(i+1))
                currLevel = currLevel+1;
            end
        end
    end
    % to obtain nodes angles array
    nodeAngles = zeros(size(treeData.Parent));
    nodeProp = zeros(size(treeData.Parent));
    for ii = 2:2:(treeData.NumNodes-1)
        % first angle, alpha
        nodeProp(ii) = 1-treeData.NodeProbability(ii)/treeData.NodeProbability(treeData.Parent(ii));
        nodeAngles(ii) = nodeProp(ii)*pi/2;
        % second angle, beta
        nodeProp(ii+1) = 1-nodeProp(ii);
        nodeAngles(ii+1) = nodeAngles(ii)-pi/2;
    end
    nodeParents = treeData.Parent;
    nodeProp = 1-nodeProp;

    treesStruc(k,1) = {nodeLevels};
    treesStruc(k,2) = {nodeAngles};
    treesStruc(k,3) = {nodeProp};
    treesStruc(k,4) = {nodeParents};
    
end

end