% calculation:
% training set: 170*25*0.7+7=2982 testing set: 170*25*0.3+3=1278
% in this project first 25 of CUM PIE is selected
clear;clc;
train_set=zeros(2982,1024);test_set=zeros(1278,1024);
load('chosenClass.mat');
for i=1:25
    randsample=randperm(170);
    for j=1:119
        train_set((i-1)*119+j,:)=reshape(im2double(imread(['PIE/',num2str(class(i)),'/',num2str(randsample(j)),'.jpg'])),1,1024);
    end
    for j=120:170
        test_set((i-1)*51+j-119,:)=reshape(im2double(imread(['PIE/',num2str(class(i)),'/',num2str(randsample(j)),'.jpg'])),1,1024);
    end
end
% load selfie
for i=1:7
    selfie=imresize(rgb2gray(imread(['PIE/selfie/',num2str(i),'.jpg'])),[32 32]);
    train_set(2975+i,:)=reshape(im2double(selfie),1,1024);
end  
for i=1:3
    selfie=imresize(rgb2gray(imread(['PIE/selfie/',num2str(i+7),'.jpg'])),[32 32]);
    test_set(1275+i,:)=reshape(im2double(selfie),1,1024);
end
