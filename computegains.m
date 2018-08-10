% see http://www.maxwell.vrac.puc-rio.br/7587/7587_4.PDF

% Author: Cássio M. M. Pereira <cassiomartini@gmail.com>
% 01/10/2015

function [ gains, attorder ] = computegains( T, y )
% Compute information gains (in order) and return
% the sort order of most useful attribute (highest information gain)
% to the lowest.

    classEnt = shannonEntropy(y);

    % compute entropy for attributes:
    attents = zeros(size(T,2), 1);
    gains = zeros(size(T,2), 1);

    for i = 1:size(T,2)
        attents(i) = attributeEntropy(T{:,i}, y);
        gains(i) = classEnt - attents(i);
    end
    
    [~, I] = sort(gains);
    attorder = I(end:-1:1);
end

