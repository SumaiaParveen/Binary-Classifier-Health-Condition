# Prediction of Colposcopy Result from Demographic, Habitual and Medical Records

## Objective

The purpose of this analysis is to predict whether a woman’s Colposcopy result would be positive or negative with the help of a machine learning model.

## Methodology

Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## Exploratory Data Analysis
See the detail here =====> [EDA]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/EDA.ipynb)

### Bivariate Relationship between the features

![Bivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/hin_bi.JPG)

### Multivariate Relationship between the features

![Multivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/hin_multi.JPG)

## Data Preprocessing
See the detail here =====> [Data Preprocessing]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/Part1_hinselmann_Data%20Preprocessing.ipynb)

### Missing Data
![ missingness]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/missing.JPG)

Different imputation techniques have been applied and finally we imputed the missing data with randomly sampled data of that particular column.

## Handling Class Imbalance 
See the detail here =====> [Class Imbalance]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/Part2_hinselmann_Handling_Class_Imb.ipynb)

Recall Score in the Minor Class Oversampled Dataset:

```
Recall Score: 0.3333
Macro Average of Recall Score: 0.5605
Weighted Average of Recall Score: 0.7814
```

Recall Score in the Synthetically Generated (SMOTE) Dataset:
```
Recall Score: 0.0 
Macro Average of Recall Score: 0.4764 
Weighted Average of Recall Score: 0.9395 
```

Recall Score in the Synthetically Generated SMOTETomek(0.95) Dataset:
```
Recall Score: 0.0
Macro Average of Recall Score: 0.4811
Weighted Average of Recall Score: 0.9488
```
Recall Score without changing anything in the dataset:
```
Recall Score: 1.0
Macro Average of Recall Score: 0.7783
Weighted Average of Recall Score: 0.5628
```

We will keep the dataset unchanged.

## Feature Selection

See the detail here =====> [Feature Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/Part3_hinselmann_Feature_Selection.ipynb)

![feat](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/feat_sel.JPG)

### The above mentioned features are chosen for this analysis.

## Model Selection
See the detail here =====> [Model Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/Part4_hinselmann_Model_Selection.ipynb)

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

![easy]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/mod_sel_easy.JPG)


## 4. Outliers/Important Observation

See the detail here =====> [Outliers]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/Part5_hinselmann_Handling_Outliers.ipynb)

Performance metrics before taking care of the outliers:
```
Macro Average of Recall Score: 0.7303
Weighted Average of Recall Score: 0.6096
------------------------------------------------------
Macro Average of Precision Score: 0.5223
Weighted Average of Precision Score: 0.9716
------------------------------------------------------
Macro Average of Recall Score: 0.4232
Weighted Average of Recall Score: 0.7354

Accuracy Score: 0.6096
```
Performance metrics after taking care of the outliers:
```
Macro Average of Recall Score: 0.75
Weighted Average of Recall Score: 0.5072
------------------------------------------------------
Macro Average of Precision Score: 0.5143
Weighted Average of Precision Score: 0.9859
------------------------------------------------------
Macro Average of Recall Score: 0.3611
Weighted Average of Recall Score: 0.6578

Accuracy Score: 0.5072
```
I’ve kept the important observation unchanged in this analysis to take care of class imbalance.

## 5. Model Performance Metrics

See the detail here =====> [Final Model]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/Part6_hinselmann_EasyEnsembleClassifier-Final.ipynb)

#### Confusion Matrix: 

We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.

![CM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/cm.JPG)

#### ROC curve and Precision-Recall curve: 

![roc_pr]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/roc_pr.JPG)

#### Performance of the model on the train and test dataset:

![train test]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Colposcopy%20Result%20Prediction/images/traintest.JPG)

## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.730
Weighted Average of Recall Score: 0.6095

Macro Average of Precision Score: 0.522
Weighted Average of Precision Score: 0.972

Macro Average of F1 Score: 0.423
Weighted Average of F1 Score: 0.735

Accuracy Score of Train Set: 0.641
Accuracy Score of Test Set: 0.6095
.
F1 Score of Train Set: 0.492
F1 Score of Test Set: 0.423
```
