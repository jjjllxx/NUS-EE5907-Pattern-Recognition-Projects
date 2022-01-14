clear;clc;load('spamData.mat');MLE_mean=zeros(2,57); MLE_variance=zeros(2,57);
yClass=[sum(ytrain()==1),sum(ytrain()==0)];ML = yClass(1)/3065;
% data process, calculate ML estimation mean and variance of training set
for i=1:3065
    for j=1:57
        MLE_mean(2-ytrain(i,1),j) = MLE_mean(2-ytrain(i,1),j)+log(Xtrain(i,j)+0.1)/yClass(2-ytrain(i,1));
    end
end
for i=1:3065
    for j=1:57
        MLE_variance(2-ytrain(i,1),j)=MLE_variance(2-ytrain(i,1),j)+(log(Xtrain(i,j)+0.1)-MLE_mean(2-ytrain(i,1),j))^2/yClass(2-ytrain(i,1));       
    end
end

%calculate the probabilty with Gaussian distribution
gauss_judge_test=zeros(2,1536);
for i=1:1536
    for j=1:57
        gauss_judge_test(1,i)=gauss_judge_test(1,i)+log(normpdf(log(Xtest(i,j)+0.1),MLE_mean(1,j),MLE_variance(1,j)^0.5));
        gauss_judge_test(2,i)=gauss_judge_test(2,i)+log(normpdf(log(Xtest(i,j)+0.1),MLE_mean(2,j),MLE_variance(2,j)^0.5));
    end
end
ypred_test_2=zeros(1536,1);correct_num_test_2=0;
for i=1:1536
    ypred_test_2(i,1)=(gauss_judge_test(1,i)+log(ML)>gauss_judge_test(2,i)+log(1-ML));
    correct_num_test_2=correct_num_test_2+(ypred_test_2(i,1)==ytest(i,1));
end
error_rate_test_2=1-correct_num_test_2/1536;% calculate training error rates

%the code below does the same thing like code above, but it deal with training set
gauss_judge_train=zeros(2,3065);
for i=1:3065
    for j=1:57
        gauss_judge_train(1,i)=gauss_judge_train(1,i)+log(normpdf(log(Xtrain(i,j)+0.1),MLE_mean(1,j),MLE_variance(1,j)^0.5));
        gauss_judge_train(2,i)=gauss_judge_train(2,i)+log(normpdf(log(Xtrain(i,j)+0.1),MLE_mean(2,j),MLE_variance(2,j)^0.5));
    end
end
ypred_train_2=zeros(1536,1);correct_num_train_2=0;
for i=1:3065
    ypred_train_2(i,1)=(gauss_judge_train(1,i)+log(ML)>gauss_judge_train(2,i)+log(1-ML));
    correct_num_train_2=correct_num_train_2+(ypred_train_2(i,1)==ytrain(i,1));
end
error_rate_train_2=1-correct_num_train_2/3065;% calculate training error rates
