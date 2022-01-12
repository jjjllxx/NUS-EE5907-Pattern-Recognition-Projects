clear;clc;load('spamData.mat');
K_value=[1:9,10:5:100];Xtrain_log=zeros(3065,57);Xtest_log=zeros(1536,57);
% data processing, transform testing set and training set into log-form
for i=1:3065
    for j=1:57
        Xtrain_log(i,j)=log(Xtrain(i,j)+0.1);
    end
end
for i=1:1536
    for j=1:57
        Xtest_log(i,j)=log(Xtest(i,j)+0.1);
    end
end

correct_num_test_4=zeros(1,28);error_rate_test_4=zeros(1,28);row=1;dist_list_test=zeros(1536,3065);
for i=1:1536
    for j=1:3065
        dist_list_test(i,j)=sum((Xtest_log(i,:)-Xtrain_log(j,:)).^2);
    end
end %calculate the distance between test and train respectively
for k=K_value
    for i = 1:1536
        dist_list_sorted_test=sort(dist_list_test(i,:));neighbors=0;  
        for num=1:k
            location=find(dist_list_test(i,:)==dist_list_sorted_test(num));
            neighbors=neighbors+ytrain(location(1),1);
        end
        correct_num_test_4(1,row)=correct_num_test_4(1,row)+(ytest(i,1)==(neighbors>=k/2));
    end
    error_rate_test_4(1,row)=1-correct_num_test_4(1,row)/1536;
    row=row+1;
end
KNN_test_1_10_100=[error_rate_test_4(1),error_rate_test_4(10),error_rate_test_4(28)];
% calculate training error rates for α = 1, 10 and 100.
figure(4)
plot(K_value, error_rate_test_4,'LineWidth',3)

%the code below does the same thing like code above, but it deal with training set
correct_num_train_4=zeros(1,28);error_rate_train_4=zeros(1,28);row=1;dist_list_train=zeros(3065,3065);
for i=1:3065
    for j=1:3065
        dist_list_train(i,j)=sum((Xtrain_log(i,:)-Xtrain_log(j,:)).^2);
    end
end
for k=K_value
    for i = 1:3065
        dist_list_sorted_train=sort(dist_list_train(i,:));neighbors=0;  
        for num=1:k
            location=find(dist_list_train(i,:)==dist_list_sorted_train(num));
            neighbors=neighbors+ytrain(location(1),1);
        end
        correct_num_train_4(1,row)=correct_num_train_4(1,row)+(ytrain(i,1)==(neighbors>=k/2));
    end
    error_rate_train_4(1,row)=1-correct_num_train_4(1,row)/3065;
    row=row+1;
end
KNN_train_1_10_100=[error_rate_train_4(1),error_rate_train_4(10),error_rate_train_4(28)];
% calculate testing error rates for α = 1, 10 and 100.
hold on
plot(K_value, error_rate_train_4,'LineWidth',3)
xlabel('K value of K-Nearest Neighbors');ylabel('Error Rate');title('Error Rate of K-Nearest Neighbors');
legend('testing','training')