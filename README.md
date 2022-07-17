# **Cardiovascular Risk Prediction**

This project is the third capstone project we've done in our [Almabetter](https://almabetter.com) Data science curriculam. All the models we have developed in this project will be heavily based on classification since we have to predict a binary dependent variable, which in this case is **10YearCHD**. It depicts if the patient is at risk or not.

## **Introduction**

Cardiovascular diseases, also called CVDs, are the leading cause of death globally, causing an estimated 17.9 million deaths each year.

CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.

The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol.

The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity.

## **Objective**
The classification goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD) based on their health statistics and information about their tobacco usage.

## **Data Gathering and description**

The dataset is available at [kaggle](https://www.kaggle.com/).

*The dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The dataset provides the patient's information and health stats. It includes over 4,000 records and 15 attributes.*

**This Dataset consists of following Attributes**:

* id: Patient identification number.

**Demographic**:

* Sex: male or female("M" or "F")
* Age: Age of the patient;(Continuous - Although the recorded ages have been truncated to
whole numbers, the concept of age is continuous)

**Behavioral**

* is_smoking: whether or not the patient is a current smoker ("YES" or "NO")
* Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be
considered continuous as one can have any number of cigarettes, even half a cigarette.)

**Medical(history)**

* BP Meds: whether or not the patient was on blood pressure medication (Nominal)
* Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)
* Prevalent Hyp: whether or not the patient was hypertensive (Nominal)
* Diabetes: whether or not the patient had diabetes (Nominal)

**Medical(current)**

* Tot Chol: total cholesterol level (Continuous)
* Sys BP: systolic blood pressure (Continuous)
* Dia BP: diastolic blood pressure (Continuous)
* BMI: Body Mass Index (Continuous)
* Heart Rate: heart rate (Continuous - In medical research, variables such as heart rate though in
fact discrete, yet are considered continuous because of large number of possible values.)
* Glucose: glucose level (Continuous)

**Dependent variable (desired target)**

* **10-year risk of coronary heart disease CHD(binary: “1”, means “Yes”, “0” means “No”)**

## **Python Libraries used**

**Datawrangling and manipulation:** 
* Numpy
* Pandas

**Visualization:** 
* Matplotib
* Seaborn 

**Machine learning Models:**
* Scikit-Learn
* tensorflow
* xgboost

**Class Imbalancy:**
* imblearn(SMOTE)

**Others:**
* warnings
* collections

## **Models**

In this project we are implementing 8 machine learning algorithms to predict the target variable and also we'll apply optimization techniques to get the best resulting accuracy.

**Following models have been used for predictions:-**

* Logistic Regression Classifier
* K-Nearest Neighbors(KNN Classifier)
* Naive Bayes Classifier
* Support Vector Machine(SVM Classifier)
* XGB Classifier
* Decision Tree Classifier
* Random Forest Classifier
* Neural Networks Classification

## **Notable Findings**

* We've noticed that XBG Classifier is the stand out performer among most of the models with an f1-score of __0.828__. It is by far the second highest score we have achieved. Hence, its safe to say that XGB Classifier provide a optimal solution to our problem. In case of Logistic regression, We were able to see the maximum f1-score of __0.656__, also in case of K-Nearest Neighbors, the f1-score extends upto __0.742__. 

* Naive Bayes Classifier showed a balanced result amongst the model we have implemented, it has a f1-score of __0.781__, which is neutral with regards to our observations across various models. But in case of SVM(Support Vector Machines) Classifier, the f1-score lies around __0.64__, which also happens to be the lowest score among all models we've implemented.

* The Random Forest Classifier has provided an optimal solution towards achieving our Objective. We were able to achieve an f1-score of __0.838__ for the test split, which is higher than any other model(excluding NN). We also noticed that in the case of Decision-tree Classifier, we were able to achieve an f1-score of __0.768__ for the test split.

* We have implemented a experimental neural network, however the results were ambigious and non-conclusive. Currently we were able to see an accuracy as high as **0.850**, but the accuracy fluctuates within a continous range.

**Finally, we can conclude that Random forest and XGB classifier might provide the best results for this particular problem, moreever we can optimise these models using Grid Search CV(cross validation) and hyperparameter tuning to get better results.**

## **Contributors**

|Name(Github)    |  Email   | 
|---------|-----------------|
|[Arvind Krishna](https://github.com/Arvind-krishna) |     killerdude.arvind@gmail.com    |
|[Keshav Sharma](https://github.com/Keshav1506) |    keshav1506sharma@gmail.com    |
|[Jayesh Panchal](https://github.com/Jayesh-Panchal) |     jaypan290497@gmail.com    |
|[Sahil Ahuja](https://github.com/saahilahujaa) |     s.ahuja38@gmail.com    |
