clc;clear;close all;
load('test_set.mat');load('train_set.mat');sample_num=500;

% random sample 493 photos plus 7 selfie
randsample=randperm(size(train_set,1)-7,sample_num-7);
PCA_train=zeros(sample_num-7,1024);
for i=1:sample_num-7
    PCA_train(i,:)=train_set(randsample(i),:);
end
PCA_train=[PCA_train;train_set(2976:2982,:)];
randsample=[randsample,2976:2982];

% calculate eigenvectors
x_means=mean(PCA_train);ss=zeros(1024);
for i=1:sample_num
    ss=ss+(PCA_train(i,:)-x_means)'*(PCA_train(i,:)-x_means)/sample_num;
end
[uu, ~, ~]=svd(ss);

% dimension=2
dim=2;
eigenvectors2=uu(:,1:dim);
project2d=zeros(sample_num,dim);
for i=1:sample_num
    project2d(i,:)=eigenvectors2'*(PCA_train(i,:)-x_means)';
end
% set color for each class
RGB_triplets=zeros(3,sample_num);palette=zeros(26,3);
for i=1:26
    palette(i,:)=[rand(1) rand(1) rand(1)];
end
for i=1:sample_num
    RGB_triplets(:,i)=palette(floor((randsample(i)-1)/119)+1,:);
end
% draw 2d projection
figure(1)
h1=scatter(project2d(494:500,1),project2d(494:500,2),100,RGB_triplets(:,494:500)','r','pentagram','filled');
hold on
scatter(project2d(1:493,1),project2d(1:493,2),30,RGB_triplets(:,1:493)','filled')
title('PCA for 2d projected data visualization')
legend(h1(1),'selfie')
hold off

% dimension=3
dim=3;
eigenvectors3=uu(:,1:dim);
project3d=zeros(sample_num,dim);
for i=1:sample_num
    project3d(i,:)=eigenvectors3'*(PCA_train(i,:)-x_means)';
end
% draw 3d projection
figure(2)
h2=scatter3(project3d(494:500,1),project3d(494:500,2),project3d(494:500,3),100,RGB_triplets(:,494:500)','r','pentagram','filled');
hold on
scatter3(project3d(1:493,1),project3d(1:493,2),project3d(1:493,3),30,RGB_triplets(:,1:493)','filled')
title('PCA for 3d projected data visualization')
legend(h2(1),'selfie')

% draw 3 corresponding eigenfaces
figure(3);title('three eigenfaces')
for i=1:3
    subplot(1,3,i)
    imshow(mat2gray(reshape(uu(:,i),32,32)))
    title(['eigenfaces',num2str(i)])
end


% dimension=40,80,200
train_num=size(train_set,1);test_num=size(test_set,1);right_num=zeros(2,3);ss_new=zeros(1024);
x_means_new=mean(train_set);
for i=1:train_num
    ss_new=ss_new+(train_set(i,:)-x_means_new)'*(train_set(i,:)-x_means_new)/train_num;
end
[uu_new, ~, ~]=svd(ss_new);dim_list=[40,80,200];

% calculate the accuracy of test_set
for k=1:3
    dim=dim_list(k);
    eigenvectors=uu_new(:,1:dim);project=zeros(train_num,dim);
    for i=1:train_num
        project(i,:)=eigenvectors'*(train_set(i,:)-x_means_new)';
    end
    for i=1:test_num
        projected=eigenvectors'*(test_set(i,:)-x_means_new)';
        distance=sum((projected'-project(1,:)).^2);type=1;
        for j=2:train_num
            if distance>sum((projected'-project(j,:)).^2)
                distance=sum((projected'-project(j,:)).^2);type=j;
            end
        end
        if floor((type-1)/119)==floor((i-1)/51)
            right_num(floor(i/1276)+1,k)=right_num(floor(i/1276)+1,k)+1;
        end  
    end
end
accuracy=zeros(2,3);accuracy(1,:)=right_num(1,:)/1275;accuracy(2,:)=right_num(2,:)/3;
