# NUS-EE5907-Project1&2
NUS Pattern Recognition course: project1 for SPAM, project2 for face recognition.

## Project1: Spam  
Each Matlab files includes one Machine Learning method, consisting of Beta-binomial Naive Bayes, Gaussian Naive Bayes, Logistic regression, K-Nearest Neighbors. The detail of each requirement is shown in EE5907_EE5027_SPAM.pdf.The raw data of spam is in spamData.mat.  

## Original README  
This is EE5907 Programming Assignment.   
1. Assignment1 to Assignment4 indicates Beta-binomial Naive Bayes,Gaussian Naive Bayes, Logistic regression and K-Nearest Neighbors respectively.    
2. Before running any code, spamData.mat should be placed in the same folder with them.    
3. The results of Assignment1,3,4 also includes a picture of training and testing error rate, which are also included in this folder,named by their assignment. 
4. Assignment4 may take about 1min to run, thank you for your time and patience.   
Jin Lexuan 23.Sept.2021

## Beta-binomial Naive Bayes
<img width="466" alt="image" src="https://user-images.githubusercontent.com/60777462/203875131-801cfe0f-86e0-4100-8bcb-67bbbdb2422b.png">

## Gaussian Naive Bayes
In Gaussian Naive Bayes question, ML estimation of mean and variance of training set are calculated for testing set and training to fit in.From the Matlab code, it can be calculated that training error rate and testing error rate are 0.1657 and 0.1602 respectively.

## Logistic regression
<img width="464" alt="image" src="https://user-images.githubusercontent.com/60777462/203875161-e0c7184b-a6a1-45b9-8f3a-d4165d7a70d4.png">

## K-Nearest Neighbors 
<img width="482" alt="image" src="https://user-images.githubusercontent.com/60777462/203875192-8a8d56a4-bf5c-42ab-b9d6-9c1fba35565c.png">



## Project2: Face Recognition   
Includes PCA,LDA,SVM,GMM,CNN algorithm.

## Original README  
This is the second assignment of EE5907. 
1. For each part, they are seperated in folders with name of the content.And PIE is the original face image and my selfie.  
2. The information of images is saved as .mat file and can be loaded automatically in each part. The original selfie are saved in 'PIE/selfie'. 
PCA,LDA and SVM part are completed in Matlab, which can be run directly. Since LIBSVM is not suitable for Mac system with M1 core, Matlab function fitcsvm is applied for SVM part.  
3. GMM and CNN are completed by python, which should be imported with required library before running. GMM is based on sklearn,while CNN is based on tensorflow.   
4. Each code need about 20s to run, CNN require few minutes to train its model. Then results will be shown. They are also saved as .fig or .png files in each folder respectively, which are convienent to see without running the code.  



