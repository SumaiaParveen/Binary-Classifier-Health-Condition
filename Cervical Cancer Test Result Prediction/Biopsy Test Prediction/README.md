# Prediction of Biopsy Result from Demographic, Habitual and Medical Records

## Objective

The purpose of this analysis is to predict whether a woman’s biopsy result would be positive or negative with the help of a machine learning model.

## Methodology

Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## Exploratory Data Analysis
See the detail here =====> [EDA]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/EDA.ipynb)

### Bivariate Relationship between the features

![Bivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/biopsy_bi.JPG)

### Multivariate Relationship between the features

![Multivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/biopsy_multi.JPG)

## Data Preprocessing
See the detail here =====> [Data Preprocessing]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/Part1_biopsy_Preprocessing.ipynb)

### Missing Data
![ missingness]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/missing.JPG)

Different imputation techniques have been applied and finally we imputed the missing data with randomly sampled data of that particular column.

## Handling Class Imbalance 
See the detail here =====> [Class Imbalance](https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/Part2_biopsy_Handling_Class_Imb.ipynb)

Recall Score in the Minor Class Oversampled Dataset:

```
Recall Score: 0.2222
Macro Average of Recall Score: 0.5529
Weighted Average of Recall Score: 0.8558
```

Recall Score in the Synthetically Generated (SMOTE) Dataset:
```
Recall Score: 0.0
Macro Average of Recall Score: 0.4709
Weighted Average of Recall Score: 0.9023
```

Recall Score in the Synthetically Generated SMOTETomek(0.95) Dataset:
```
Recall Score: 0.0
Macro Average of Recall Score: 0.4854
Weighted Average of Recall Score: 0.9302
```
Recall Score without changing anything in the dataset:
```
Recall Score: 0.8889
Macro Average of Recall Score: 0.709
Weighted Average of Recall Score: 0.5442
```

We will keep the dataset unchanged.

## Feature Selection

See the detail here =====> [Feature Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/Part3_biopsy_Feature_Selection.ipynb)

![feat]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/feat_sel.png)

### Selected Features:

+ ‘age', 
+ 'hormonalcontraceptives(years)', 
+ 'first_sexual_intercourse', 
+ 'number_of_sexual_partners', 
+ 'num_ofpregnancies',
+  'iud(years)', 
+ 'smokes(years)', 
+ 'smokes(packs_year)', 
+ 'hormonal_contraceptives', 
+ 'iud', 
+ 'dx_cancer', 
+ 'dx_hpv', 
+ 'dx', 
+ 'stds_number_ofdiagnosis',
+  'stds(number)'



## Model Selection
See the detail here =====> [Model Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/Part4_biopsy_Model_Selection.ipynb)

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

![easy]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/pred_err.JPG)


## 4. Outliers/Important Observation

See the detail here =====> [Outliers]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/Part5_biopsy_Handling_Outliers.ipynb)

Performance metrics before taking care of the outliers:
```
Recall Score: 1.0
Macro Average of Recall Score: 0.7925
Weighted Average of Recall Score: 0.6065
------------------------------------------------------
Macro Average of Precision Score: 0.558
Weighted Average of Precision Score: 0.9544
------------------------------------------------------
Macro Average of Recall Score: 0.473
Weighted Average of Recall Score: 0.7108
------------------------------------------------------
Accuracy Score: 0.6065
```
Performance metrics after taking care of the outliers:
```
Recall Score: 0.875
Macro Average of Recall Score: 0.7368
Weighted Average of Recall Score: 0.6129
------------------------------------------------------
Macro Average of Precision Score: 0.5474
Weighted Average of Precision Score: 0.9432
------------------------------------------------------
Macro Average of Recall Score: 0.4675
Weighted Average of Recall Score: 0.717
------------------------------------------------------
Accuracy Score: 0.6129
```
I’ve kept the important observation unchanged in this analysis.

## 5. Model Performance Metrics

See the detail here =====> [Final Model]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/Part6_biopsy_EasyEnsembleClassifier-Final.ipynb)

#### Confusion Matrix: 

We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.

![CM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/cm.JPG)

#### ROC curve and Precision-Recall curve: 

![roc_pr]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/roc_pr.JPG)

#### Performance of the model on the train and test dataset:

![train test]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Biopsy%20Test%20Prediction/images/traintest.JPG)

## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.793
Weighted Average of Recall Score: 0.606

Macro Average of Precision Score: 0.558
Weighted Average of Precision Score: 0.954

Macro Average of F1 Score: 0.473
Weighted Average of F1 Score: 0.711

Accuracy Score of Train Set: 0.623
Accuracy Score of Test Set: 0.606

F1 Score of Train Set: 0.501
F1 Score of Test Set: 0.473
```
