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

## 1. Data Preprocessing
See the detail here =====> [Preprocessing]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/Part1_crossdementia_Preprocessing.ipynb)

### Missing Data
![Missingness of Dataset]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/missing.JPG)
Almost 50% of the data are missing in the ‘SES” column and we want to keep the importance of missingness and thus we imputed ‘0’ if any data was missing in this particular column.
## . Model Selection
See the detail here =====> [Model Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/Part2_crossdementia_Model_Selection.ipynb)

### Selected Model: 
CatBoostClassifier(random_state = 42)

```
{'depth': 6,
 'iterations': 1000,
 'l2_leaf_reg': 5,
 'leaf_estimation_iterations': 10,
 'learning_rate': 0.015,
 'logging_level': 'Silent',
 'loss_function': 'Logloss',
 'random_seed': 18}

```
3. ## Handling Class Imbalance 
See the detail here =====> [Class Imbalance]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/Part3_crossdementia_Handling_Class_Imbalance.ipynb)
The minority class have been oversampled in this dataset. See the images below. 
![Original]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/regular.JPG)

![Oversampled]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/oversampled.JPG)
## 4. Outliers/Important Observation
See the detail here =====> [Outliers]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/Part4_crossdementia_Handling_Outliers.ipynb)
Performance metrics before taking care of the outliers:
```
Recall Score: 0.8889
Macro Average of Recall Score: 0.899
Weighted Average of Recall Score: 0.9
------------------------------------------------------
Macro Average of Precision Score: 0.899
Weighted Average of Precision Score: 0.9
------------------------------------------------------
Macro Average of F1 Score: 0.899
Weighted Average of F1 Score: 0.9
------------------------------------------------------
Accuracy Score: 0.9

```
Performance metrics after taking care of the outliers:
```
Recall Score: 0.8889
Macro Average of Recall Score: 0.899
Weighted Average of Recall Score: 0.9
------------------------------------------------------
Macro Average of Precision Score: 0.899
Weighted Average of Precision Score: 0.9
------------------------------------------------------
Macro Average of F1 Score: 0.899
Weighted Average of F1 Score: 0.9
------------------------------------------------------
Accuracy Score: 0.9

```
We’ve kept the important observation unchanged in this analysis.

## Model Performance Metrics
See the detail here =====> [Final Model]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/Part5_crossdementia_CatBoostClassifier-Final.ipynb)

#### Confusion Matrix: 
We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.
![CM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/cm.JPG)
#### ROC curve and Precision-Recall curve: 
![roc_pr]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/roc_pr.JPG)
#### Performance of the model on the train and test dataset:
![train test]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/traintest.JPG)
## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.899
Weighted Average of Recall Score: 0.9

Macro Average of Precision Score: 0.899
Weighted Average of Precision Score: 0.9

Macro Average of F1 Score: 0.899
Weighted Average of F1 Score: 0.9

Accuracy Score of Train Set: 1.0
Accuracy Score of Test Set: 0.9

F1 Score of Train Set: 1.0
F1 Score of Test Set: 0.899

```
