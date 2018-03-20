k=10;
newData = csvread('dataset.csv');
newDataset = [newData(:,1) newData(:,2) newData(:,3)];
newGroup = newData(:,4);
%newDataset = Dataset;
%newGroup = Group;
cvFolds = crossvalind('Kfold', newGroup, k);   %# get indices of 10-fold CV
cp = classperf(newGroup);                      %# init performance tracker
%dataset = [Dataset(:,1) Dataset(:,2)];
dataset = newDataset;
metode = 'SMO';
tic;
for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    %# train an SVM model over training instances
    svmModel = svmtrain(dataset(trainIdx,:), newGroup(trainIdx), ...
                 'Autoscale',true, 'Showplot',false, 'Method',metode, ...
                 'BoxConstraint',2e-1, 'Kernel_Function','rbf','rbf_sigma',1);

    %# test using test instances
    pred = svmclassify(svmModel, dataset(testIdx,:), 'Showplot',false);

    %# evaluate and update performance object
    cp = classperf(cp, pred, testIdx);
end

%# get accuracy
SMOSIGMA1 = cp.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOSIGMA1mat = cp.CountingMatrix
toc;
tic;
cp2 = classperf(newGroup);
for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    %# train an SVM model over training instances
    svmModel = svmtrain(dataset(trainIdx,:), newGroup(trainIdx), ...
                 'Autoscale',true, 'Showplot',false, 'Method',metode, ...
                 'BoxConstraint',2e-1, 'Kernel_Function','linear');

    %# test using test instances
    pred = svmclassify(svmModel, dataset(testIdx,:), 'Showplot',false);

    %# evaluate and update performance object
    cp2 = classperf(cp2, pred, testIdx);
end

%# get accuracy
SMOlinear = cp2.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOlinearmat = cp2.CountingMatrix
toc;
tic;
cp3 = classperf(newGroup);
for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    %# train an SVM model over training instances
    svmModel = svmtrain(dataset(trainIdx,:), newGroup(trainIdx), ...
                 'Autoscale',true, 'Showplot',false, 'Method',metode, ...
                 'BoxConstraint',2e-1, 'Kernel_Function','quadratic');

    %# test using test instances
    pred = svmclassify(svmModel, dataset(testIdx,:), 'Showplot',false);

    %# evaluate and update performance object
    cp3 = classperf(cp3, pred, testIdx);
end

%# get accuracy
SMOquadratic = cp3.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOquadraticmat = cp3.CountingMatrix
toc;
tic;
cp4 = classperf(newGroup);
for i = 1:k                                  %# for each fold
    testIdx = (cvFolds == i);                %# get indices of test instances
    trainIdx = ~testIdx;                     %# get indices training instances

    %# train an SVM model over training instances
    svmModel = svmtrain(dataset(trainIdx,:), newGroup(trainIdx), ...
                 'Autoscale',true, 'Showplot',false, 'Method',metode, ...
                 'BoxConstraint',2e-1, 'Kernel_Function','polynomial');

    %# test using test instances
    pred = svmclassify(svmModel, dataset(testIdx,:), 'Showplot',false);

    %# evaluate and update performance object
    cp4 = classperf(cp4, pred, testIdx);
end

%# get accuracy
SMOpoly = cp4.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOpolymat = cp4.CountingMatrix
toc;