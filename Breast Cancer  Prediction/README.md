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

! https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Strawberry_ice_cream_cone_%285076899310%29.jpg/220px-Strawberry_ice_cream_cone_%285076899310%29.jpg
