a = VectorFiturNew(VectorFiturNew(:, 4) == 1, :);
b = VectorFiturNew(VectorFiturNew(:, 4) == 0, :);
a_smote = smote(a, 200, 5);
training = [b(1:243, :);a_smote(1:82, :)];
testing = [b(244:364, :);a_smote(83:123, :)];
train = training(:,1:3);
groupTrain = training(:,4);
tes = testing(:,1:3);
groupTes = testing(:,4);
cp = classperf(groupTes);                      %# init performance tracker
metode = 'SMO';
tic;
%# train an SVM model over training instances
svmModel = svmtrain(train, groupTrain, ...
             'Autoscale',true, 'Showplot',false, 'Method',metode, ...
             'BoxConstraint',2e-1, 'Kernel_Function','rbf','rbf_sigma',1);

%# test using test instances
pred = svmclassify(svmModel, tes, 'Showplot',false);

%# evaluate and update performance object
cp = classperf(cp, pred);

%# get accuracy
SMOSIGMA1 = cp.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOSIGMA1mat = cp.CountingMatrix
SMOsigmaRecall = SMOSIGMA1mat(2,2) / (SMOSIGMA1mat(2,2) + SMOSIGMA1mat(2,1))
SMOsigmaPrecision = SMOSIGMA1mat(2,2) / (SMOSIGMA1mat(2,2) + SMOSIGMA1mat(1,2))
toc;
tic;
cp2 = classperf(groupTes);

%# train an SVM model over training instances
svmModel = svmtrain(train, groupTrain, ...
             'Autoscale',true, 'Showplot',false, 'Method',metode, ...
             'BoxConstraint',2e-1, 'Kernel_Function','linear');

%# test using test instances
pred = svmclassify(svmModel, tes, 'Showplot',false);

%# evaluate and update performance object
cp2 = classperf(cp2, pred);

%# get accuracy
SMOlinear = cp2.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOlinearmat = cp2.CountingMatrix
SMOlinearRecall = SMOlinearmat(2,2) / (SMOlinearmat(2,2) + SMOlinearmat(2,1))
SMOlinearPrecision = SMOlinearmat(2,2) / (SMOlinearmat(2,2) + SMOlinearmat(1,2))
toc;
tic;
cp3 = classperf(groupTes);
    
%# train an SVM model over training instances
svmModel = svmtrain(train, groupTrain, ...
             'Autoscale',true, 'Showplot',false, 'Method',metode, ...
             'BoxConstraint',2e-1, 'Kernel_Function','quadratic');

%# test using test instances
pred = svmclassify(svmModel, tes, 'Showplot',false);

%# evaluate and update performance object
cp3 = classperf(cp3, pred);


%# get accuracy
SMOquadratic = cp3.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOquadraticmat = cp3.CountingMatrix
SMOquadraticRecall = SMOquadraticmat(2,2) / (SMOquadraticmat(2,2) + SMOquadraticmat(2,1))
SMOquadraticPrecision = SMOquadraticmat(2,2) / (SMOquadraticmat(2,2) + SMOquadraticmat(1,2))
toc;
tic;
cp4 = classperf(groupTes);
    
%# train an SVM model over training instances
svmModel = svmtrain(train, groupTrain, ...
             'Autoscale',true, 'Showplot',false, 'Method',metode, ...
             'BoxConstraint',2e-1, 'Kernel_Function','polynomial');

%# test using test instances
pred = svmclassify(svmModel, tes, 'Showplot',false);

%# evaluate and update performance object
cp4 = classperf(cp4, pred);

%# get accuracy
SMOpoly = cp4.CorrectRate

%# get confusion matrix
%# columns:actual, rows:predicted, last-row: unclassified instances
SMOpolymat = cp4.CountingMatrix
SMOpolyRecall = SMOpolymat(2,2) / (SMOpolymat(2,2) + SMOpolymat(2,1))
SMOpolyPrecision = SMOpolymat(2,2) / (SMOpolymat(2,2) + SMOpolymat(1,2))
toc;