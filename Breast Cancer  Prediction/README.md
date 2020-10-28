# Breast Cancer Prediciton from Breast Mass Features

## Objective
The purpose of this analysis is to predict the class of breast cancer (either benign or malice) by employing a Machine Learning model. The data set was created and made open by Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian in the General Surgery and Computer Science Departments as the University of Wisconsin. 

## About the Dataset
Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

The last 30 features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. 

Attribute information
1. ID number
2. Diagnosis (M = malignant, B = benign)

Feature (3-32): Ten real-valued features are computed for each cell nucleus:

1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 4 is Mean Radius, field 14 is Radius SE, field 24 is Worst Radius.
Values for features 4-33 are recoded with four significant digits.

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

## Some numerics values we'd like to look at

+ Macro Average of Recall Score: 0.985
+ Weighted Average of Recall Score: 0.986
.
+ Macro Average of Precision Score: 0.985
+ Weighted Average of Precision Score: 0.986
.
+ Macro Average of F1 Score: 0.985
+ Weighted Average of F1 Score: 0.986
.
+ Accuracy Score of Train Set: 1.0
+ Accuracy Score of Test Set: 0.986
.
+ F1 Score of Train Set: 1.0
+ F1 Score of Test Set: 0.985









