# ML-algorithm-comparison-in-Matlab

The following project examined the performance and efficacy of two machine learning algorithms, namely Random Forest and Naïve Bayes, in a multi-class classification task, determining level of obesity as the target variable based on lifestyle and physiology data, obtained from the UCI Machine Learning Repository (linked and referenced below).

The bulk of the programming was conducted in Matlab. Synthetic data produced by SMOTE technology was omitted to lower accuracy in both algorithms to generate points of comparison (test accuracy for both optimised algorithms when using the whole data set was >99.5%).

The files are structured as follows: pre-processing; separate RF and NB files that explore different versions of the prepped data (e.g. dummy variables or no dummy variables), as well as different hyperparameter settings using grid-search. K-fold Cross validation is used as the main means of validation in this part of model selection. Finally, there are two .m files containing the best-performing model for each algorithm, which are loaded and executed on the test data in two separate final_model.mat files. Performance metrics and confisions matrices are calcualated here to provide some post-processing analysis. Most evaluation and post-processing is displayed in the poster.pdf file and supplementary materials pdf file.

data set: https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+

from:

Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico.
