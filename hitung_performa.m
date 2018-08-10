matrix = zeros(2,2);
matrix(1,1) = sum(unnamed(unnamed(:,1)==0,1)==unnamed(unnamed(:,1)==0,2));
matrix(2,1) = sum(unnamed(unnamed(:,1)==0,1)~=unnamed(unnamed(:,1)==0,2));
matrix(1,2) = sum(unnamed(unnamed(:,1)==1,1)~=unnamed(unnamed(:,1)==1,2));
matrix(2,2) = sum(unnamed(unnamed(:,1)==1,1)==unnamed(unnamed(:,1)==1,2));

svm_conv_mat = matrix;
akurasi = (svm_conv_mat(1,1) + svm_conv_mat(2,2)) / sum(sum(svm_conv_mat))
precision = svm_conv_mat(2,2) / (svm_conv_mat(2,2) + svm_conv_mat(2,1))
tp_rate = svm_conv_mat(2,2) / (svm_conv_mat(2,2) + svm_conv_mat(1,2))
tn_rate = svm_conv_mat(1,1) / (svm_conv_mat(1,1) + svm_conv_mat(2,1))
f1_meas = 2*precision*tp_rate / (tp_rate + precision)