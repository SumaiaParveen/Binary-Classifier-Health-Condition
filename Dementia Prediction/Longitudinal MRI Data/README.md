# Dementia Prediction from Longitudinal MRI Data of Older Adults

## Objective

The purpose of this analysis is to predict whether an older adult is demented or non-demented by employing a Machine Learning model. 

## About the Dataset

The Open Access Series of Imaging Studies [(OASIS)](https://www.oasis-brains.org/) is a project aimed at making MRI data sets of the brain freely available to the scientific community. [OASIS](https://www.oasis-brains.org/) is made available by the Washington University Alzheimer’s Disease Research Center, Dr. Randy Buckner at the Howard Hughes Medical Institute (HHMI)( at Harvard University, the Neuroinformatics Research Group (NRG) at Washington University School of Medicine, and the Biomedical Informatics Research Network (BIRN).

Source: https://central.xnat.org/app/template/XDATScreen_report_xnat_projectData.vm/search_element/xnat:projectData/search_field/xnat:projectData.ID/search_value/CENTRAL_OASIS_LONG

This set consists of a longitudinal collection of 150 subjects aged 60 to 96. Each subject was scanned on two or more visits, separated by at least one year for a total of 373 imaging sessions. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. 72 of the subjects were characterized as nondemented throughout the study. 64 of the included subjects were characterized as demented at the time of their initial visits and remained so for subsequent scans, including 51 individuals with mild to moderate Alzheimer’s disease. Another 14 subjects were characterized as nondemented at the time of their initial visit and were subsequently characterized as demented at a later visit.

#### Acknowledgement: Longitudinal MRI Data in Nondemented and Demented Older Adults
Marcus, DS, Fotenos, AF, Csernansky, JG, Morris, JC, Buckner, RL, 2010. Journal of Cognitive Neuroscience, 22, 2677-2684
. doi: [10.1162/jocn.2009.21407]( https://www.mitpressjournals.org/doi/full/10.1162/jocn.2009.21407)

Attribute information

+ Subject.ID - Unique Id of the patient
+ MRI.ID - Unique Id generated after conducting MRI on patient
+ Group - It is a group of Converted (Previously Normal but developed dimentia later), Demented and Nondemented (Normal Pateints)
+ Visit - Number of visit to detect dementia status
+ M.F - Gender
+ Hand - Handedness 
+ Age - Age in years
+ EDUC - Years of education
+ SES - Socioeconomic status as assessed by the Hollingshead Index of Social Position and classified into categories from 1 (highest status) to 5 (lowest status)
+ MMSE - Mini-Mental State Examination score (range is from 0 = worst to 30 = best)
+ CDR - Clinical Dementia Rating
+ eTIV - Estimated total intracranial volume, mm3
+ nWBV - Normalized whole-brain volume, expressed as a percent of all voxels in the atlas-masked image that are labeled as gray or white matter by the automated tissue segmentation process
+ ASF - Atlas scaling factor (unitless). Computed scaling factor that transforms native-space brain and skull to the atlas target (i.e., the determinant of the transform matrix)

## Methodology
Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## 1. Data Preprocessing
See the detail here =====> [Preprocessing]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/Part1_longdementia_Preprocessing.ipynb)

### Missing Data
![Missingness of Dataset]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/images/missing.JPG)

Almost 50% of the data are missing in the ‘SES” column and we want to keep the importance of missingness and thus we imputed ‘0’ if any data was missing in this particular column. Randomly sampled data were imputed in the ‘MMSE’ column.

## 2. Model Selection
See the detail here =====> [Model Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/Part2_longdementia_Model_Selection.ipynb)

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
See the detail here =====> [Class Imbalance]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/Part3_longdementia_Handling_Class_Imbalance.ipynb)

The dataset is kept unchanged.

Recall Score in the Minor Class Oversampled Dataset:

```
Recall Score: 0.7805
Macro Average of Recall Score: 0.8336
Weighted Average of Recall Score: 0.8404
```

Recall Score in the Synthetically Generated (SMOTE) Dataset:
```
Recall Score: 0.8049
Macro Average of Recall Score: 0.8553
Weighted Average of Recall Score: 0.8617
```

Recall Score in the Regular Dataset:
```
Recall Score: 0.8292682926829268
Macro Average of Recall Score: 0.8674643350207087
Weighted Average of Recall Score: 0.8723404255319149
```

## 4. Outliers/Important Observation

See the detail here =====> [Outliers]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/Part4_longdementia_Handling_Outliers.ipynb)

Performance metrics before taking care of the outliers:
```
Recall Score: 0.8333
Macro Average of Recall Score: 0.8712
Weighted Average of Recall Score: 0.8763
------------------------------------------------------
Macro Average of Precision Score: 0.8761
Weighted Average of Precision Score: 0.8762
------------------------------------------------------
Macro Average of F1 Score: 0.8733
Weighted Average of F1 Score: 0.8759
------------------------------------------------------
Accuracy Score: 0.8763
```
Performance metrics after taking care of the outliers:
```
Recall Score: 0.8333
Macro Average of Recall Score: 0.8712
Weighted Average of Recall Score: 0.8763
------------------------------------------------------
Macro Average of Precision Score: 0.8761
Weighted Average of Precision Score: 0.8762
------------------------------------------------------
Macro Average of F1 Score: 0.8733
Weighted Average of F1 Score: 0.8759
------------------------------------------------------
Accuracy Score: 0.8763
```
I’ve kept the important observation unchanged in this analysis.

## 5. Model Performance Metrics

See the detail here =====> [Final Model]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/Part5_longdementia_CatBoostClassifier-Final.ipynb)

#### Confusion Matrix: 

We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.

![CM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/images/cm.JPG)

#### ROC curve and Precision-Recall curve: 

![roc_prhttps://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/images/roc_pr.JPG)

#### Performance of the model on the train and test dataset:

![train test]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Dementia%20Prediction/Longitudinal%20MRI%20Data/images/traintest.JPG)

## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.892
Weighted Average of Recall Score: 0.897

Macro Average of Precision Score: 0.897
Weighted Average of Precision Score: 0.897

Macro Average of F1 Score: 0.894
Weighted Average of F1 Score: 0.897

Accuracy Score of Train Set: 1.0
Accuracy Score of Test Set: 0.897

F1 Score of Train Set: 1.0
F1 Score of Test Set: 0.89
```
