clear;clc;load('spamData.mat');naive_bayes = zeros(2,57);
for i = 1:3065
    for j = 1:57
        naive_bayes(2-ytrain(i,1),j) = naive_bayes(2-ytrain(i,1),j)+(Xtrain(i,j) > 0);
    end
end %to calculate data numbers of ytrain=1 and ytrain=0 respectively, for each feature

yClass=[sum(ytrain()==1),sum(ytrain()==0)];ML = yClass(1)/3065;
isSpam_test = zeros(201,1536);isnotSpam_test = zeros(201,1536);
for alpha = 0:200
    for i = 1:1536
        for j = 1:57
            if Xtest(i,j) > 0 
                isSpam_test(alpha+1,i) = isSpam_test(alpha+1,i)+log((naive_bayes(1,j)+alpha/2)/(yClass(1)+alpha));
                isnotSpam_test(alpha+1,i) = isnotSpam_test(alpha+1,i)+log((naive_bayes(2,j)+alpha/2)/(yClass(2)+alpha));
            else
                isSpam_test(alpha+1,i) = isSpam_test(alpha+1,i)+log(1-(naive_bayes(1,j)+alpha/2)/(yClass(1)+alpha));
                isnotSpam_test(alpha+1,i) = isnotSpam_test(alpha+1,i)+log(1-(naive_bayes(2,j)+alpha/2)/(yClass(2)+alpha));
            end
        end
    end
end  %to calculate the probality of ML estimate with α

correct_num_test_1 = zeros(1,201);error_rate_test_1 = zeros(1,201);
for alpha = 1:201
    ypred_test_1 = zeros(1536,1);
    for i = 1:1536
        ypred_test_1(i,1) = (isSpam_test(alpha,i)+log(ML)>isnotSpam_test(alpha,i)+log(1-ML)); 
        correct_num_test_1(1,alpha) = correct_num_test_1(1,alpha)+(ypred_test_1(i,1)==ytest(i,1));
    end
    error_rate_test_1(1,alpha) = (1-correct_num_test_1(1,alpha)/1536); 
end % do prediction and calculate error rate
alpha_test_1_10_100=[error_rate_test_1(3),error_rate_test_1(21),error_rate_test_1(201)];alpha = 0:0.5:100;
figure(1)
plot(alpha,error_rate_test_1,'LineWidth',3)
xlabel('Hyperparameter for Beta prior');ylabel('Error Rate');title('Error Rate of Beta-binomial Naive Bayes');

%the code below does the same thing like code above, but it deal with training set
isSpam_train = zeros(201,3065);isnotSpam_train = zeros(201,3065);
for alpha = 0:200
    for i = 1:3065
        for j = 1:57
            if Xtrain(i,j) > 0 
                isSpam_train(alpha+1,i) = isSpam_train(alpha+1,i)+log((naive_bayes(1,j)+alpha/2)/(yClass(1)+alpha));
                isnotSpam_train(alpha+1,i) = isnotSpam_train(alpha+1,i)+log((naive_bayes(2,j)+alpha/2)/(yClass(2)+alpha));
            else
                isSpam_train(alpha+1,i) = isSpam_train(alpha+1,i)+log(1-(naive_bayes(1,j)+alpha/2)/(yClass(1)+alpha));
                isnotSpam_train(alpha+1,i) = isnotSpam_train(alpha+1,i)+log(1-(naive_bayes(2,j)+alpha/2)/(yClass(2)+alpha));
            end
        end
    end
end
correct_num_train_1 = zeros(1,201);error_rate_train_1 = zeros(1,201);
for alpha = 1:201
    ypred_train_1 = zeros(3065,1);
    for i = 1:3065
        ypred_train_1(i,1) = (isSpam_train(alpha,i)+log(ML)>isnotSpam_train(alpha,i)+log(1-ML)); 
        correct_num_train_1(1,alpha) = correct_num_train_1(1,alpha)+(ypred_train_1(i,1)==ytrain(i,1));
    end
    error_rate_train_1(1,alpha) = (1-correct_num_train_1(1,alpha)/3065); 
end
alpha_train_1_10_100=[error_rate_train_1(3),error_rate_train_1(21),error_rate_train_1(201)];
% calculate training and testing error rates for α = 1, 10 and 100.
hold on
alpha = 0:0.5:100;
plot(alpha,error_rate_train_1,'LineWidth',3)
legend('testing','training')
