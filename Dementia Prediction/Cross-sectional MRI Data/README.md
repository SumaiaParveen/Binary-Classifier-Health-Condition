# Dementia Prediction from Cross-sectional MRI Data of Young, Middle Aged Person

## Objective
The purpose of this analysis is to predict whether a young or middled aged person is demented or non-demented by employing a Machine Learning model. 

## About the Dataset

The Open Access Series of Imaging Studies [(OASIS)](https://www.oasis-brains.org/) is a project aimed at making MRI data sets of the brain freely available to the scientific community. [OASIS](https://www.oasis-brains.org/) is made available by the Washington University Alzheimer’s Disease Research Center, Dr. Randy Buckner at the Howard Hughes Medical Institute (HHMI)( at Harvard University, the Neuroinformatics Research Group (NRG) at Washington University School of Medicine, and the Biomedical Informatics Research Network (BIRN).

Source: https://central.xnat.org/app/template/XDATScreen_report_xnat_projectData.vm/search_element/xnat:projectData/search_field/xnat:projectData.ID/search_value/CENTRAL_OASIS_CS

This set consists of a cross-sectional collection of 416 subjects aged 18 to 96. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. 100 of the included subjects over the age of 60 have been clinically diagnosed with very mild to moderate Alzheimer’s disease (AD). Additionally, a reliability data set is included containing 20 nondemented subjects imaged on a subsequent visit within 90 days of their initial session.

#### Acknowledgement: Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults
Marcus, DS, Wang, TH, Parker, J, Csernansky, JG, Morris, JC, Buckner, RL. Journal of Cognitive Neuroscience, 19, 1498-1507. doi: [10.1162/jocn.2007.19.9.1498](https://www.mitpressjournals.org/doi/abs/10.1162/jocn.2007.19.9.1498)

Attribute information

+ ID: Identification
+ M/F: Gender
+ Hand: Dominant Hand
+ Age: Age in years
+ Educ: Education Level
+ SES: Socioeconomic Status
+ MMSE: Mini Mental State Examination
+ eTIV: Estimated Total Intracranial Volume
+ nWBV: Normalize Whole Brain Volume
+ ASF: Atlas Scaling Factor
+ CDR: Clinical Dementia Rating

## Methodology
Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## 1. Data Preprocessing
See the detail here =====> [Preprocessing]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/Part1_crossdementia_Preprocessing.ipynb)

### Missing Data
![Missingness of Dataset]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/missing.JPG)

Almost 50% of the data are missing in the ‘SES” column and we want to keep the importance of missingness and thus we imputed ‘0’ if any data was missing in this particular column.
## 2. Model Selection
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
## 3. Handling Class Imbalance 
See the detail here =====> [Class Imbalance]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/Part3_crossdementia_Handling_Class_Imbalance.ipynb)

The minority class have been oversampled in this dataset. See the images below. 

Target Variable in the Orginal Dataset

![Original]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Cross-sectional%20MRI%20Data/images/regular.JPG)

Target Variable in the Oversampled Dataset

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
I’ve kept the important observation unchanged in this analysis.

## 5. Model Performance Metrics

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
