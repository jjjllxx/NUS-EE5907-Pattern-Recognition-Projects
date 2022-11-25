# NUS-EE5907-Pattern-Recognition-Projects
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

## PCA for Feature Extraction, Visualization and Classification
<img width="380" alt="image" src="https://user-images.githubusercontent.com/60777462/203875438-18a30b37-d550-4f99-b260-b51fb7261396.png"><img width="380" alt="image" src="https://user-images.githubusercontent.com/60777462/203875443-bc2c73aa-65ee-4ead-8497-f97ea737b766.png">

<img width="350" alt="image" src="https://user-images.githubusercontent.com/60777462/203875464-81feff9d-a493-4a5e-b900-4d4737f73051.png">

## LDA for Feature Extraction and Classification
<img width="380" alt="image" src="https://user-images.githubusercontent.com/60777462/203875577-5a534729-0b0b-49fd-8d5c-d85405910545.png"><img width="380" alt="image" src="https://user-images.githubusercontent.com/60777462/203875599-8c54f013-a30e-4f3e-b6a5-6ec956014def.png">

<img width="370" alt="image" src="https://user-images.githubusercontent.com/60777462/203875605-207b07f2-6826-4891-a0b0-7a9b7839ee11.png"><img width="390" alt="image" src="https://user-images.githubusercontent.com/60777462/203875616-47283744-6c43-4295-b6fa-a72cf1ee518b.png">

<img width="385" alt="image" src="https://user-images.githubusercontent.com/60777462/203875634-d2464094-4f17-491f-a63a-bbdaf0e21839.png"><img width="375" alt="image" src="https://user-images.githubusercontent.com/60777462/203875652-eb5a4183-9306-4ee5-bf65-5260f087111e.png">

## GMM for clustering
### Raw image
<img width="480" alt="image" src="https://user-images.githubusercontent.com/60777462/203876147-5943ac69-5322-46d1-9460-c303773db359.png">

<img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876179-0a01398d-58ff-4955-b1bb-7c4bed9958d5.png"><img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876168-415d471c-f3f4-455a-8e04-fcfab0911e84.png"><img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876187-3f8d5fc9-68b7-4841-99ed-0ff7324b0486.png">

### PCA = 80
<img width="480" alt="image" src="https://user-images.githubusercontent.com/60777462/203876505-b062dc74-c586-4474-8113-b9a000e1cc36.png">

<img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876508-51079ab7-36ed-4331-8842-52d8b70e6197.png"><img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876520-38f9f4ec-79bb-4c08-bb8d-d5708c1cdfc0.png"><img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876534-6f945d7a-2927-4878-abd7-f6b0745c4ad4.png">

### PCA = 200
<img width="480" alt="image" src="https://user-images.githubusercontent.com/60777462/203876674-6b86b713-f004-498b-9b71-009867d625ca.png">

<img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876687-1aac7074-703e-4c7f-848c-e3dd630d7555.png"><img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876683-76899d53-8343-471c-b5c6-cf2b9c8180b2.png"><img width="250" alt="image" src="https://user-images.githubusercontent.com/60777462/203876693-52ed09e5-de8e-41ce-9a5f-69c7b9c6c187.png">

## SVM for Classification

| C parameter| raw image|	PCA 80 | PCA 200|
|  ----  | ----  |  ----  | ----  |
| C=0.01	| 98.44% |92.88%	| 96.09% |
|C=0.1	|98.75%	| 95.38%	|97.18% |
| C=1	 |98.83%	|96.40%	| 97.89% |

## Neural Networks for Classification
<img width="380" alt="image" src="https://user-images.githubusercontent.com/60777462/203877316-acc4f5bc-9230-45df-822f-ccd549308c41.png"><img width="380" alt="image" src="https://user-images.githubusercontent.com/60777462/203877270-39fd3dbb-90c2-4ba7-8ef7-e35fcedee45d.png">
