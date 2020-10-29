# Parkinson's Disease Prediction from Speech Features

## Objective

This analysis is a part of some attempts to develop and become skilled in binary classification using machine learning techniques. I have tried to build a model with highest possible (according to my experiments) Recall score in order to classify any given data/dataset as Parkinson’s Disease (PD) or not. In future, there will be more attempts to employ Deep Learning models and understand PCA intuitively using this dataset. 

## About the Dataset

Source:  https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification#
The data used in this study were gathered from 188 patients with PD (107 men and 81 women) with ages ranging from 33 to 87 at the Department of Neurology in Cerrahpasa Faculty of Medicine, Istanbul University. The control group consists of 64 healthy individuals (23 men and 41 women) with ages varying between 41 and 82. During the data collection process, the microphone is set to 44.1 KHz and following the physician’s examination, the sustained phonation of the vowel ‘a’ was collected from each subject with three repetitions.
Attribute Information:

The following features have been applied to the speech recordings of Parkinson's Disease (PD) patients to extract clinically useful information for PD assessment.
+ Various speech signal processing algorithms including 
+ Time Frequency Features, 
+ Mel Frequency Cepstral Coefficients (MFCCs), 
+ Wavelet Transform based Features, 
+ Vocal Fold Features and 
+ TWQT


Acknowledgement:  Sakar, B.E., Tutuncu, M., Aydin, T., Isenkul, M.E. and Apaydin, H., 2018. A comparative analysis of speech signal processing algorithms for Parkinson’s
disease classification and the use of the tunable Q-factor wavelet transform. Applied Soft Computing, DOI: [https://doi.org/10.1016/j.asoc.2018.10.022](https://www.sciencedirect.com/science/article/abs/pii/S1568494618305799?via%3Dihub)

## Methodology

Several machine learning algorithms are applied to see which model works better on this particular dataset. Before that, 15 important features are selected after applying five feature selection/ feature importance methods. Outlier/ important observation and their impact have also been taken care of.

## Model Selection
See the detail here =====> [Model Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/Part2_parkinsons_Model_Selection.ipynb)

### Comparison between XGBoost and LightGBM Model

Prediction error using XGBoost model:

![XGB]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/mod_sel_xgb.JPG)

Prediction error using LightGBM model:

![LGBM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/mod_sel_lgbm.JPG)

### Selected Model: 
LGBMClassifier(random_state = 42)

```
{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 100,
 'n_jobs': -1,
 'num_leaves': 31,
 'objective': None,
 'random_state': 42,
 'reg_alpha': 0.0,
 'reg_lambda': 0.0,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0}
```
## Handling Class Imbalance 
See the detail here =====> [Class Imbalance]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/Part3_parkinsons_Handling_Class_Imbalance.ipynb)

Recall Score in the Minor Class Oversampled Dataset:

```
Recall Score: 0.9653
Macro Average of Recall Score: 0.8382
Weighted Average of Recall Score: 0.9048
```

Recall Score in the Synthetically Generated (SMOTE) Dataset:
```
Recall Score: 0.9375
Macro Average of Recall Score: 0.8688
Weighted Average of Recall Score: 0.9048
```

Recall Score in the Synthetically Generated SMOTETomek(0.95) Dataset:
```
Recall Score: 0.9306
Macro Average of Recall Score: 0.8653
Weighted Average of Recall Score: 0.8995
```
We will oversample the minor class.

Target variable in the original dataset:

![regular]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/regular.JPG)

Target variable in the oversampled dataset:

![os]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/os.JPG)

## Feature Selection

See the detail here =====> [Feature Selection]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/Part4_parkisons_Feature_Selection.ipynb)

![feat]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/feat_sel.JPG)

### Selected Features:

+ 'tqwt_entropy_log_dec_33', 
+ 'tqwt_minvalue_dec_25', 
+ 'tqwt_entropy_shannon_dec_25', 
+ 'tqwt_meanvalue_dec_36', 
+ 'tqwt_kurtosisvalue_dec_33', 
+ 'tqwt_kurtosisvalue_dec_31', 
+ 'tqwt_maxvalue_dec_17', 
+ 'std_9th_delta', 
+ 'tqwt_maxvalue_dec_13', 
+ 'tqwt_entropy_log_dec_28', 
+ 'tqwt_tkeo_std_dec_19', 
+ 'tqwt_meanvalue_dec_22', 
+ 'imf_snr_seo', 
+ 'tqwt_meanvalue_dec_5', 
+ 'tqwt_kurtosisvalue_dec_27'


## 4. Outliers/Important Observation

See the detail here =====> [Outliers]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/Part5_parkinsons_Handling_Outliers.ipynb)

Performance metrics before taking care of the outliers:
```
Recall Score: 0.9561
Macro Average of Recall Score: 0.7939
Weighted Average of Recall Score: 0.875
------------------------------------------------------
Macro Average of Precision Score: 0.8569
Weighted Average of Precision Score: 0.8715
------------------------------------------------------
Macro Average of F1 Score: 0.8181
Weighted Average of F1 Score: 0.869
------------------------------------------------------
Accuracy Score: 0.875

```
Performance metrics after taking care of the outliers:
```
Recall Score: 0.9126
Macro Average of Recall Score: 0.8093
Weighted Average of Recall Score: 0.8613
------------------------------------------------------
Macro Average of Precision Score: 0.8156
Weighted Average of Precision Score: 0.86
------------------------------------------------------
Macro Average of F1 Score: 0.8123
Weighted Average of F1 Score: 0.8606
------------------------------------------------------
Accuracy Score: 0.8613
```
I’ve kept the important observation unchanged in this analysis.

## 5. Model Performance Metrics

See the detail here =====> [Final Model]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/Part6_parkinsons_LGBMClassifier-Final.ipynb)

#### Confusion Matrix: 

We wanted to minimize the False Negative as small as possible and thus maximize the Recall score.

![CM]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/cm.JPG)

#### ROC curve and Precision-Recall curve: 

![roc_pr]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/roc_pr.JPG)

#### Performance of the model on the train and test dataset:

![train test]( https://github.com/SumaiaParveen/Binary-Classifier-Health-Condition/blob/main/Parkinsons%20Prediction/images/traintest.JPG)

## Some numeric values we'd like to look at

```
Macro Average of Recall Score: 0.794
Weighted Average of Recall Score: 0.875
.
Macro Average of Precision Score: 0.857
Weighted Average of Precision Score: 0.872

Macro Average of F1 Score: 0.818
Weighted Average of F1 Score: 0.869
.
Accuracy Score of Train Set: 1.0
Accuracy Score of Test Set: 0.875
.
F1 Score of Train Set: 1.0
F1 Score of Test Set: 0.818
```
