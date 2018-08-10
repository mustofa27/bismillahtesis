k=10;
prosentase_kelas_positif = 0.4;
%newData = csvread('dataset.csv');
%newData = VectorFiturTranslate;
%newData = VectorFitur;
%a = VectorFiturNew(VectorFiturNew(:, 4) == 1, :);
%b = VectorFiturNew(VectorFiturNew(:, 4) == 0, :);
fitur = cell2mat(cosine(:,3:5));
kelas = cell2mat(cosine(:,8));
VectorFiturNew = [fitur,kelas];
%newData = [a_smote;b];
newData = VectorFiturNew;
newDataset = [newData(:,1) newData(:,2) newData(:,3)];
newGroup = newData(:,4);
%newDataset = [newData(:,1) newData(:,2) newData(:,3) newData(:,4) newData(:,5)];
%newGroup = newData(:,6);
hasil = zeros(size(newGroup));
%newDataset = Dataset;
%newGroup = Group;
cvFolds = crossvalind('Kfold', newGroup, k);   %# get indices of 10-fold CV
cp = classperf(newGroup);                      %# init performance tracker
%dataset = [Dataset(:,1) Dataset(:,2)];
dataset = newDataset;
metode = 'SMO';
tic;
cp4 = classperf(newGroup);

for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    %# train an SVM model over training instances
    train = [dataset(trainIdx,:) newGroup(trainIdx)];
    a = train(train(:, 4) == 1, :);
    dimen_all = size(train);
    dimen = size(a);
    smote_size = int64(prosentase_kelas_positif * dimen_all(1) * 100 / dimen(1)) - 100;
    a_smote = smote(a, smote_size, 10);
    train_final = [train;a_smote];
    svmModel = svmtrain(train_final(:,1:3), train_final(:,4), ...
                 'Autoscale',true, 'Showplot',false, 'Method',metode, ...
                 'BoxConstraint',2e-1, 'Kernel_Function','polynomial');

    %# test using test instances
    tes = [dataset(testIdx,:) newGroup(testIdx)];
    a_tes = tes(tes(:, 4) == 1, :);
    dimen_all_tes = size(tes);
    dimen_tes = size(a_tes);
    smote_size_tes = int64(prosentase_kelas_positif * dimen_all_tes(1) * 100 / dimen_tes(1)) - 100;
    a_smote_tes = smote(a_tes, smote_size_tes, 20);
    tes_final = [tes;a_smote];
    pred = svmclassify(svmModel, tes_final(:,1:3), 'Showplot',false);
    hasil(testIdx,:) = pred(1:dimen_all_tes,:);
    %# evaluate and update performance object
    cp4 = classperf(cp4, pred(1:dimen_all_tes,:), testIdx);
end

%# get accuracy
SMOpoly = cp4.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
svm_conv_mat = cp4.CountingMatrix
svm_precision = svm_conv_mat(2,2) / (svm_conv_mat(2,2) + svm_conv_mat(2,1))
svm_tp_rate = svm_conv_mat(2,2) / (svm_conv_mat(2,2) + svm_conv_mat(1,2))
svm_tn_rate = svm_conv_mat(1,1) / (svm_conv_mat(1,1) + svm_conv_mat(2,1))
svm_f1_meas = 2*svm_precision*svm_tp_rate / (svm_tp_rate + svm_precision)
toc;