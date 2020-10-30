# Prediction of Cytology Result from Demographic, Habitual and Medical Records

## Objective

The purpose of this analysis is to predict whether a woman’s Cytology result would be positive or negative with the help of a machine learning model.

## About the Dataset

Source:  https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29 

The dataset was collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela. The dataset comprises demographic information, habits, and historic medical records of 858 patients. Several patients decided not to answer some of the questions because of privacy concerns (Missing not at Random).

Attribution Information:

+ Age
+ Number of sexual partners
+ First sexual intercourse (age)
+ Num of pregnancies
+ Smokes
+ Smokes (years)
+ Smokes (packs/year)
+ Hormonal Contraceptives
+ Hormonal Contraceptives (years)
+ IUD
+ IUD (years)
+ STDs
+ STDs (number)
+ STDs: condylomatosis
+ STDs: cervical condylomatosis
+ STDs: vaginal condylomatosis
+ STDs: vulvo-perineal condylomatosis
+ STDs: syphilis
+ STDs: pelvic inflammatory disease
+ STDs: genital herpes
+ STDs: molluscum contagiosum
+ STDs: AIDS
+ STDs: HIV
+ STDs: Hepatitis B
+ STDs: HPV
+ STDs: Number of diagnosis
+ STDs: Time since first diagnosis
+ STDs: Time since last diagnosis
+ Dx: Cancer
+ Dx: CIN
+ Dx: HPV
+ Dx
+ Cytology: target variable

Acknowledgement:  Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.' Iberian Conference on Pattern Recognition and Image Analysis. Springer International Publishing, 2017.

## Methodology

Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## Exploratory Data Analysis
See the detail here =====> [EDA]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/EDA.ipynb)

### Bivariate Relationship between the features

![Bivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/cyto_bi.JPG)

### Multivariate Relationship between the features

![Multivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/cyto_multi.JPG)

## Data Preprocessing
See the detail here =====> [Data Preprocessing]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/Part1_citology_Data%20Preprocessing.ipynb)

### Missing Data
![ missingness]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/missing.JPG)

Different imputation techniques have been applied and finally we imputed the missing data with randomly sampled data of that particular column.

## Handling Class Imbalance 
See the detail here =====> [Class Imbalance]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/Part2_citology_Handling_Class_Imb.ipynb)

Recall Score in the Minor Class Oversampled Dataset:

```
Recall Score: 0.0
Macro Average of Recall Score: 0.4802
Weighted Average of Recall Score: 0.9023
```

Recall Score in the Synthetically Generated (SMOTE) Dataset:
```

Recall Score: 0.0
Macro Average of Recall Score: 0.4802
Weighted Average of Recall Score: 0.9023
```

Recall Score in the Synthetically Generated SMOTETomek(0.95) Dataset:
```
Recall Score: 0.0
Macro Average of Recall Score: 0.4901
Weighted Average of Recall Score: 0.9209
```
Recall Score without changing anything in the dataset:
```
Recall Score: 0.3846
Macro Average of Recall Score: 0.4794
Weighted Average of Recall Score: 0.5628
```

We will keep the dataset unchanged.

## Feature Selection

See the detail here =====> [Feature Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/Part3_citology_Feature_Selection.ipynb)

![feat]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/feat_sel.JPG)

### Selected Features:

+ ‘number_of_sexual_partners', 
+ 'first_sexual_intercourse', 
+ 'hormonalcontraceptives(years)', 
+ 'age', 
+ 'smokes', 
+ 'smokes_(packs_year)', 
+ 'stds_number_ofdiagnosis',
+ 'smokes(years)',
+ 'dx_hpv', 
+ 'dxcancer',
+ 'iud(years)', 
+ 'iud', 
+ 'num_of_pregnancies', 
+ 'dx', 
+ 'stds_syphilis'

## Model Selection
See the detail here =====> [Model Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/Part4_citology_Model_Selection.ipynb)

### Selected Model: 

EasyEnsembleClassifier(random_state = 25)

```
{'base_estimator': None,
 'n_estimators': 10,
 'n_jobs': None,
 'random_state': 25,
 'replacement': False,
 'sampling_strategy': 'auto',
 'verbose': 0,
 'warm_start': False}
```
Prediction error using EasyEnsembleClassifier model:

![easy]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/pred_err.JPG)


## 4. Outliers/Important Observation

See the detail here =====> [Outliers](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/Part5_citology_Handling_Outliers.ipynb)

Performance metrics before taking care of the outliers:
```
Macro Average of Recall Score: 0.5833
Weighted Average of Recall Score: 0.5116
------------------------------------------------------
Macro Average of Precision Score: 0.5216
Weighted Average of Precision Score: 0.8923
------------------------------------------------------
Macro Average of Recall Score: 0.4079
Weighted Average of Recall Score: 0.6212

Accuracy Score: 0.5116
```
Performance metrics after taking care of the outliers:
```
Macro Average of Recall Score: 0.5792
Weighted Average of Recall Score: 0.5039
------------------------------------------------------
Macro Average of Precision Score: 0.5206
Weighted Average of Precision Score: 0.8915
------------------------------------------------------
Macro Average of Recall Score: 0.4031
Weighted Average of Recall Score: 0.6141

Accuracy Score: 0.5039
```
I’ve kept the important observation unchanged in this analysis to take care of class imbalance.

## 5. Model Performance Metrics

See the detail here =====> [Final Model]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/Part6_citology_EasyEnsembleClassifier-Final.ipynb)

#### Confusion Matrix: 

We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.

![CM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/cm.JPG)

#### ROC curve and Precision-Recall curve: 

![roc_pr]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/roc_pr.JPG)

#### Performance of the model on the train and test dataset:

![train test]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Cytology%20Result%20Prediction/images/traintest.JPG)

## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.583
Weighted Average of Recall Score: 0.512

Macro Average of Precision Score: 0.522
Weighted Average of Precision Score: 0.892

Macro Average of F1 Score: 0.408
Weighted Average of F1 Score: 0.621

Accuracy Score of Train Set: 0.534
Accuracy Score of Test Set: 0.512

F1 Score of Train Set: 0.423
F1 Score of Test Set: 0.408
```
