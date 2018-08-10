% Author: Cássio M. M. Pereira <cassiomartini@gmail.com>
% 01/10/2015

%clear all; clc;

rng default;

x = unifrnd(1,10,20,1);
y = [repmat({'abc'}, 10, 1) ; repmat({'def'}, 10, 1)];
classe = [repmat({'class1'}, 10, 1) ; repmat({'class2'}, 10, 1)];

T = table(x, y, classe);

[gains, attorder] = computegains(T(:,[1 2]), T{:,3});

