# Prediction of Schiller’s Test Result from Demographic, Habitual and Medical Records

## Objective

The purpose of this analysis is to predict whether a woman’s cervical cancer test result would be positive or negative with the help of a machine learning model.

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
+ Schiller: target variable

Acknowledgement:  Kelwin Fernandes, Jaime S. Cardoso, and Jessica Fernandes. 'Transfer Learning with Partial Observability Applied to Cervical Cancer Screening.' Iberian Conference on Pattern Recognition and Image Analysis. Springer International Publishing, 2017.

## Methodology

Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## Exploratory Data Analysis
See the detail here =====> [EDA]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/EDA.ipynb)

### Bivariate Relationship between the features

![Bivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/schi_bi.JPG)

### Multivariate Relationship between the features

![Multivariate]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/schi_multi.JPG)

## Data Preprocessing
See the detail here =====> [Data Preprocessing]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/Part1_schiller_Data%20Preprocessing.ipynb)

### Missing Data
![ missingness]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/missing.JPG)

Different imputation techniques have been applied and finally we imputed the missing data with randomly sampled data of that particular column.

## Handling Class Imbalance 
See the detail here =====> [Class Imbalance]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/Part2_schiller_Handling_Class_Imb.ipynb)

Recall Score in the Minor Class Oversampled Dataset:

```
Recall Score: 0.4
Macro Average of Recall Score: 0.561
Weighted Average of Recall Score: 0.707
```

Recall Score in the Synthetically Generated (SMOTE) Dataset:
```

Recall Score: 0.1
Macro Average of Recall Score: 0.5329
Weighted Average of Recall Score: 0.9256
```

Recall Score in the Synthetically Generated SMOTETomek(0.95) Dataset:
```
Recall Score: 0.0
Macro Average of Recall Score: 0.4659
Weighted Average of Recall Score: 0.8884

```
Recall Score without changing anything in the dataset:
```
Recall Score: 0.6
Macro Average of Recall Score: 0.6073
Weighted Average of Recall Score: 0.614
```

We will keep the dataset unchanged.

## Feature Selection

See the detail here =====> [Feature Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/Part3_schiller_Feature_Selection.ipynb)

![feat]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/feat_sel.JPG)

### The above features are chosen for this analysis.

## Model Selection
See the detail here =====> [Model Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/Part4_schiller_Model_Selection.ipynb)

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
Prediction error using EasyEnsemble Classifier model:

![easy]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/mod_easy.JPG)

Prediction error using GaussianNB Classifier model:

![nb]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/mod_nb.JPG)


## 4. Outliers/Important Observation

See the detail here =====> [Outliers]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/Part5_schiller_Handling_Outliers.ipynb)

Performance metrics before taking care of the outliers:
```
Macro Average of Recall Score: 0.651
Weighted Average of Recall Score: 0.5942
------------------------------------------------------
Macro Average of Precision Score: 0.5297
Weighted Average of Precision Score: 0.9295
------------------------------------------------------
Macro Average of Recall Score: 0.4424
Weighted Average of Recall Score: 0.7038

Accuracy Score: 0.5942
```
Performance metrics after taking care of the outliers:
```
Macro Average of Recall Score: 0.6778
Weighted Average of Recall Score: 0.6449
------------------------------------------------------
Macro Average of Precision Score: 0.5364
Weighted Average of Precision Score: 0.9321
------------------------------------------------------
Macro Average of Recall Score: 0.4718
Weighted Average of Recall Score: 0.7435

Accuracy Score: 0.6449
```
I’ve kept the important observation unchanged in this analysis to take care of class imbalance.

## 5. Model Performance Metrics

See the detail here =====> [Final Model]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/Part6_schiller_EasyEnsembleClassifier-Final.ipynb)

#### Confusion Matrix: 

We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.

![CM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/cm.JPG)

#### ROC curve and Precision-Recall curve: 

![roc_pr]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/roc_pr.JPG)

#### Performance of the model on the train and test dataset:

![train test]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Cervical%20Cancer%20Test%20Result%20Prediction/Schiller's%20Test%20Result%20Prediction/images/traintest.JPG)

## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.678
Weighted Average of Recall Score: 0.645

Macro Average of Precision Score: 0.536
Weighted Average of Precision Score: 0.932

Macro Average of F1 Score: 0.472
Weighted Average of F1 Score: 0.744

Accuracy Score of Train Set: 0.668
Accuracy Score of Test Set: 0.645

F1 Score of Train Set: 0.558
F1 Score of Test Set: 0.472
```
