clc;clear;close;
load('test_set.mat');load('train_set.mat');
C=[0.01,0.1,1];correct_rate=zeros(3);
train_num=size(train_set,1);test_num=size(test_set,1);
classes=1:26;class_num=length(classes);
SVM_model = cell(class_num,1);

% raw face image
for k=1:3
    % train SVM model
    for i=1:class_num
        label_tmp=zeros(train_num,1);
        for j=(i-1)*119+1:min(i*119,train_num)
            label_tmp(j,1)=1;
        end
        SVM_model{i}=fitcsvm(train_set,label_tmp,'Standardize',true,'KernelFunction','linear','BoxConstraint',C(k));
    end
    test_num=size(test_set,1);
    all_score = zeros(test_num,class_num);
    % verify test set
    for i=1:class_num
        [~,score] = predict(SVM_model{i},test_set);
        all_score(:,i) = score(:,2);
    end
    correct_num=0;
    for i=1:test_num
        if find(all_score(i,:)==max(all_score(i,:)))==floor((i-1)/51)+1
            correct_num=correct_num+1;
        end
    end
    correct_rate(1,k)=correct_num/test_num;
end

% PCA dimensionality reduction
ss_new=zeros(1024);x_means_new=mean(train_set);
for i=1:train_num
    ss_new=ss_new+(train_set(i,:)-x_means_new)'*(train_set(i,:)-x_means_new)/train_num;
end
[uu_new, ~, ~]=svd(ss_new);

% PCA 40 and PCA 80
for dim=[40,80]
    % PCA dimensionality reduction
    eigenvectors=uu_new(:,1:dim);
    PCA_train=zeros(train_num,dim);PCA_test=zeros(test_num,dim);
    for i=1:train_num
        PCA_train(i,:)=eigenvectors'*(train_set(i,:)-x_means_new)';
    end
    for i=1:test_num
        PCA_test(i,:)=eigenvectors'*(test_set(i,:)-x_means_new)';
    end
    for k=1:3
        % train SVM model
        for i=1:class_num
            label_tmp=zeros(train_num,1);
            for j=(i-1)*119+1:min(i*119,train_num)
                label_tmp(j,1)=1;
            end
            SVM_model{i}=fitcsvm(PCA_train,label_tmp,'Standardize',true,'KernelFunction','linear','BoxConstraint',C(k),'IterationLimit',150);
        end
        test_num=size(PCA_test,1);
        all_score = zeros(test_num,class_num);
        % verify test set
        for i=1:class_num
            [~,score] = predict(SVM_model{i},PCA_test);
            all_score(:,i) = score(:,2);
        end
        correct_num=0;
        for i=1:test_num
            if find(all_score(i,:)==max(all_score(i,:)))==floor((i-1)/51)+1
                correct_num=correct_num+1;
            end
        end
        correct_rate(dim/40+1,k)=correct_num/test_num;
    end
end
