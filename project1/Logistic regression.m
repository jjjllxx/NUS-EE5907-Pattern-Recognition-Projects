clear;clc;load('spamData.mat');
regularizion_parameter=[1:9,10:5:100];% l2 regularization
% add bias term
Xtrain_log=[ones(3065,1),Xtrain];Xtest_log=[ones(1536,1),Xtest];
omega_regularization=zeros(28,58);
% data processing, transform testing set and training set into log-form
for i=1:3065
    for j=2:58
        Xtrain_log(i,j)=log(Xtrain_log(i,j)+0.1);
    end
end
for i=1:1536
    for j=2:58
        Xtest_log(i,j)=log(Xtest_log(i,j)+0.1);
    end
end
u=zeros(3065,1);row=1;
% Newtow Method iteration
for lambda=regularizion_parameter
    omega=zeros(58,1);
    for k=1:50
        for i=1:3065
            u(i,1)=1./(1+exp(-Xtrain_log(i,:)*omega));
        end
        g=Xtrain_log'*(u-ytrain)+lambda.*[0,omega(2:58,1)']';%gradient
        S=diag((u.*(1.-u))');
        h=Xtrain_log'*S*Xtrain_log+lambda.*diag([0,ones(1,57)]);%hessian
        omega=omega-h^(-1)*g; %calculate new omega
    end
    omega_regularization(row,:)=omega';
    row=row+1;
end
correct_num_test_3=zeros(1,28);error_rate_test_3=zeros(1,28);
for lambda=1:28
    for i=1:1536
        y_pred_3=1./(1+exp(-omega_regularization(lambda,:)*Xtest_log(i,:)'));
        correct_num_test_3(1,lambda)=correct_num_test_3(1,lambda)+(ytest(i,1)==(y_pred_3>0.5));
    end
    error_rate_test_3(1,lambda)=1-correct_num_test_3(1,lambda)/1536;
end
rp_test_1_10_100=[error_rate_test_3(1),error_rate_test_3(10),error_rate_test_3(28)];
% calculate testing error rates for α = 1, 10 and 100.
figure(3)
plot(regularizion_parameter,error_rate_test_3,'LineWidth',3)

%the code below does the same thing like code above, but it deal with training set
correct_num_train_3=zeros(1,28);error_rate_train_3=zeros(1,28);
for lambda=1:28
    for i=1:3065
        y_pred_3=1./(1+exp(-omega_regularization(lambda,:)*Xtrain_log(i,:)'));
        correct_num_train_3(1,lambda)=correct_num_train_3(1,lambda)+(ytrain(i,1)==(y_pred_3>0.5));
    end
    error_rate_train_3(1,lambda)=1-correct_num_train_3(1,lambda)/3065;
end
rp_train_1_10_100=[error_rate_train_3(1),error_rate_train_3(10),error_rate_train_3(28)];
% calculate training error rates for α = 1, 10 and 100.
hold on
plot(regularizion_parameter,error_rate_train_3,'LineWidth',3)
xlabel('Regularization Parameter of Logistic Regression');ylabel('Error Rate');title('Error Rate of Logistic Regression');
legend('testing','training')
