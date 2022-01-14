clc;clear;close all;
% the number of each class
load('test_set.mat');load('train_set.mat');
train_num=size(train_set,1);test_num=size(test_set,1);

sample_num=493; % sample_num can be different(493,1193,2975...)

% random sample face photos plus 7 selfie
randsample=randperm(train_num-7,sample_num);LDA_train=zeros(sample_num,1024);
for i=1:sample_num
    LDA_train(i,:)=train_set(randsample(i),:);
end
LDA_train=[LDA_train;train_set(2976:2982,:)];randsample=[randsample,2976:2982];
class_num=zeros(1,26);uc=zeros(26,1024);ut=mean(LDA_train);
class=floor((randsample-1)/119)+1;
for i=1:sample_num+7
    class_num(class(i))=class_num(class(i))+1;
    uc(class(i),:)=uc(class(i),:)+LDA_train(i,:);
end
uc=uc./class_num';

% calculate the variance between class
ssb=zeros(1024,1024);ssw=zeros(1024,1024);
for i=1:26
    ssb=ssb+class_num(i)*(uc(i,:)-ut)'*(uc(i,:)-ut);
end
% calculate the variance within class
for i=1:sample_num+7
    ssw=ssw+(LDA_train(i,:)-uc(class(i)))'*(LDA_train(i,:)-uc(class(i)));
end
[ww,~,~]=svd(pinv(ssw)*ssb);

% set color for each class
RGB_triplets=zeros(3,sample_num);palette=zeros(26,3);
for i=1:26
    palette(i,:)=[rand(1) rand(1) rand(1)];
end
for i=1:sample_num
    RGB_triplets(:,i)=palette(class(i),:);
end

% plot 2d LDA
draw2=LDA_train*ww(:,1:2);
figure(1)
h1=scatter(draw2(sample_num+1:sample_num+7,1),draw2(sample_num+1:sample_num+7,2),100,'r','pentagram','filled');
hold on
scatter(draw2(1:sample_num,1),draw2(1:sample_num,2),30,RGB_triplets(:,1:sample_num)','filled')
title(['LDA for 2d projected data visualization(sample:' num2str(sample_num+7),')'])
legend(h1(1),'selfie')
hold off

% plot 3d LDA
draw3=LDA_train*ww(:,1:3);
figure(2)
h2=scatter3(draw3(sample_num+1:sample_num+7,1),draw3(sample_num+1:sample_num+7,2),draw3(sample_num+1:sample_num+7,3),100,'r','pentagram','filled');
hold on
scatter3(draw3(1:sample_num,1),draw3(1:sample_num,2),draw3(1:sample_num,3),30,RGB_triplets(:,1:sample_num)','filled')
title(['LDA for 3d projected data visualization(sample:' num2str(sample_num+7),')'])
legend(h2(1),'selfie')
hold off

% calculate accuracy
% calculate the variance between class
mean_t=mean(train_set);mean_c=zeros(26,1024);class_num=[ones(1,25)*119,7];
for i=1:26
    mean_c(i,:)=mean(train_set(119*(i-1)+1:119*(i-1)+class_num(i),:));
end
SSB=zeros(1024,1024);SSW=zeros(1024,1024);
for i=1:26
    SSB=SSB+class_num(i)*(mean_c(i,:)-mean_t)'*(mean_c(i,:)-mean_t);
end

% calculate the variance within class
for i=1:26
    for j=119*(i-1)+1:119*(i-1)+class_num(i)
        SSW=SSW+(train_set(j,:)-mean_c(i))'*(train_set(j,:)-mean_c(i));
    end
end
[WW,~,~]=svd(pinv(SSW)*SSB);

% calculate the accuracy of test_set
dim=[2,3,9];right_num=zeros(2,3);test=zeros(3,1278);
for k=1:3
    project=train_set*WW(:,1:dim(k));
    for i=1:test_num
        project1=test_set(i,:)*WW(:,1:dim(k));
        distance=sum((project1-project(1,:)).^2);type=1;
        for j=2:train_num
            if distance>sum((project1-project(j,:)).^2)
                distance=sum((project1-project(j,:)).^2);type=j;
            end
        end
        test(k,i)=type;
        if floor((i-1)/51)==floor((type-1)/119)
            right_num(floor(i/1276)+1,k)=right_num(floor(i/1276)+1,k)+1;  
        end
    end
end
accuracy=zeros(2,3);
accuracy(1,:)=right_num(1,:)/1275;accuracy(2,:)=right_num(2,:)/3;
