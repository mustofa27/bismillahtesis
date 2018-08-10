k=10;
%newData = csvread('dataset.csv');
%newData = VectorFiturTranslate;
%newData = VectorFitur;
fitur = cell2mat(cosine(:,3:5));
kelas = cell2mat(cosine(:,8));
VectorFiturNew = [fitur,kelas];
a = VectorFiturNew(VectorFiturNew(:, 4) == 1, :);
b = VectorFiturNew(VectorFiturNew(:, 4) == 0, :);
a_smote = smote(a, 200, 20);
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

    model = fitcknn(dataset(trainIdx,:),newGroup(trainIdx),'NumNeighbors',5);
    %# test using test instances
    pred = predict(model,dataset(testIdx,:));
    hasil(testIdx,:) = pred;
    %# evaluate and update performance object
    cp = classperf(cp, pred, testIdx);
end

%# get accuracy
knn_acc = cp.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
knn_conv_mat = cp.CountingMatrix
knn_precision = knn_conv_mat(2,2) / (knn_conv_mat(2,2) + knn_conv_mat(2,1))
knn_tp_rate = knn_conv_mat(2,2) / (knn_conv_mat(2,2) + knn_conv_mat(1,2))
knn_tn_rate = knn_conv_mat(1,1) / (knn_conv_mat(1,1) + knn_conv_mat(2,1))
knn_f1_meas = 2*knn_precision*knn_tp_rate / (knn_tp_rate + knn_precision)
toc;