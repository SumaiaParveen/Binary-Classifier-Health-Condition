# Chronic Kidney Disease Prediciton

## Objective
The purpose of this analysis is to predict the presence of chronic kidney diease by employing a Machine Learning model. L. Jerlin Rubini created the data, with the collaboration of Doctors P. Soundarapandian and P. Eswaran.

## About the Dataset
Source: https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease

The dataset has 400 rows, one per patient; these are the patients observed over a period of about two months at some point before July 2015, in a hospital in Tamil Nadu, India.

Attribute information

+ Age: age in years
+ Blood Pressure: : BP in mm/Hg (Diastolic Pressure)
+ Specific Gravity: one of (1.005,1.010,1.015,1.020,1.025)
+ Albumin: one of (0,1,2,3,4,5)
+ Sugar: one of (0,1,2,3,4,5) 
+ Red Blood Cells: either 'Normal' or 'Abnormal'
+ Pus Cell: either 'Normal' or 'Abnormal'
+ Pus Cell clumps: either 'Present' or 'Not Present'
+ Bacteria: either 'Present' or 'Not Present'
+ Blood Glucose Random: in mgs/dl
+ Blood Urea: in mgs/dl
+ Serum Creatinine: in mgs/dl
+ Sodium: in mEq/L
+ Potassium: in mEq/L
+ Hemoglobin: in gms
+ Packed Cell Volume: Volume Percentage
+ White Blood Cell Count: in cells/cumm
+ Red Blood Cell Count: in millions/cmm
+ Hypertension: either 'Yes' or 'No'
+ Diabetes Mellitus: either 'Yes' or 'No'
+ Coronary Artery Disease: either 'Yes' or 'No'
+ Appetite: either 'Good' or 'Poor'
+ Pedal Edema: either 'Yes' or 'No'
+ Anemia: either 'Yes' or 'No'
+ Class : either 'ckd'==> 'Chronic Kidney Disease' or 'notckd'==> ' NOT Chronic Kidney Disease'

To know more in detail about the dataset, please visit Matthew Brett's website [here](https://matthew-brett.github.io/cfd2019/data/chronic_kidney_disease)

## Methodology
Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## Dataset Infographics
See the detail here =====> [EDA](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/Part1_breastcancer_Preprocessin_EDA.ipynb)

### Corelation between some of the features

![Corelation between some of the features](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/images/corealtion.JPG)

### Bivariate Relationship between the features

![Bivariate](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/images/bivariate.JPG)

### Multivariate Relationship between the features

![Multivariate](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/images/multivariate.JPG)

## Selected Features
See the detail here =====> [Feature Selection](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/Part4_breastcancer_Feature_Selection.ipynb)

1.	'area_worst',
2.  'concave points_worst',
3.  'concave points_mean',
4.  'perimeter_worst',
5.	'concavity_mean',
6.	'radius_worst',
7.	'radius_mean',
8.	'perimeter_mean',
9.	'area_se',
10.	'area_mean',
11.	'concavity_worst',
12.	'compactness_worst',
13.	'texture_worst',
14.	'smoothness_worst',
15.	'radius_se'

where mean = mean value, se = standard error and worst = mean of the three largest values

## Model Performance Metrics

#### Selected Model: 
AdaBoostClassifier(random_state = 42)
See the detail here =====> [Model Selection](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/Part2_breastcancer_Model_Selection.ipynb)

#### Confusion Matrix: [Model Evaluation](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/Part6_breastcancer_AdaBoostClassifier-Final.ipynb)

We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.

![CM](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/images/cm.JPG)

#### ROC curve and Precision-Recall curve: [Final Model](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/Part6_breastcancer_AdaBoostClassifier-Final.ipynb)

![roc_pr](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/images/roc_pr.JPG)

#### Performance of the model on the train and test dataset: [Final Model](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/Part6_breastcancer_AdaBoostClassifier-Final.ipynb)


![roc_pr](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Breast%20Cancer%20%20Prediction/images/train_test.JPG)

## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.985
Weighted Average of Recall Score: 0.986

Macro Average of Precision Score: 0.985
Weighted Average of Precision Score: 0.986

Macro Average of F1 Score: 0.985
Weighted Average of F1 Score: 0.986

Accuracy Score of Train Set: 1.0
Accuracy Score of Test Set: 0.986

F1 Score of Train Set: 1.0
F1 Score of Test Set: 0.985
```









