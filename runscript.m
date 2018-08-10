% Author: Cássio M. M. Pereira <cassiomartini@gmail.com>
% 01/10/2015

clear all;
clc;

load fisheriris;

T = array2table(meas);
y = species;

[gains, attorder] = computegains(T, y);
[ranked, weight] = relieff(meas, y, 10);

subplot(1,2,1);
bar(attorder, gains(attorder));
xlabel('atributo'); ylabel('ganho de informação');

subplot(1,2,2);
bar(ranked, weight(ranked));
xlabel('atributo'); ylabel('relieff rank');
