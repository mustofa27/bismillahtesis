k=10;
%newData = csvread('dataset.csv');
%newData = VectorFiturTranslate;
%newData = VectorFitur;
fitur = cell2mat(cosine(:,3:5));
kelas = cell2mat(cosine(:,8));
VectorFiturNew = [fitur,kelas];
a = VectorFiturNew(VectorFiturNew(:, 4) == 1, :);
b = VectorFiturNew(VectorFiturNew(:, 4) == 0, :);
a_smote = smote(a, 800, 20);
%newData = [a_smote;b];
newData = VectorFiturNew;
newDataset = [newData(:,1) newData(:,2) newData(:,3)];
newGroup = newData(:,4);
%newDataset = Dataset;
%newGroup = Group;
cvFolds = crossvalind('Kfold', newGroup, k);   %# get indices of 10-fold CV
cp = classperf(newGroup);                      %# init performance tracker
%dataset = [Dataset(:,1) Dataset(:,2)];
dataset = newDataset;
metode = 'SMO';
hasil = zeros(size(newGroup));
tic;
for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    net = feedforwardnet(20);
    [net,tr] = train(net,transpose(dataset(trainIdx,:)), transpose(newGroup(trainIdx)));
    %# test using test instances
    pred = transpose(net(transpose(dataset(testIdx,:))));
    pred = abs(round(pred));
    hasil(testIdx,:) = pred;
    %# evaluate and update performance object
    cp = classperf(cp, pred, testIdx);
end

%# get accuracy
neural_network = cp.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
nn_conv_mat = cp.CountingMatrix
nn_precision = nn_conv_mat(2,2) / (nn_conv_mat(2,2) + nn_conv_mat(2,1))
nn_tp_rate = nn_conv_mat(2,2) / (nn_conv_mat(2,2) + nn_conv_mat(1,2))
nn_tn_rate = nn_conv_mat(1,1) / (nn_conv_mat(1,1) + nn_conv_mat(2,1))
nn_f1_meas = 2*nn_precision*nn_tp_rate / (nn_tp_rate + nn_precision)
toc;