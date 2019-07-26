# Contextual_Regression_for_CircRNA
## Requirements
Python3 (we used 3.64)
Tensorflow (we used v1.5)
Numpy (we used v1.14.0)
Scipy (we used v1.0.0)
Sklearn (we used v0.19.1)

## Description
This project is an implementation of contextual regression for circular RNA data, it can train and predict 
whether a region contains circular RNA based on sequence features (whether the region contains CpG islands, etc.)
in the region. It can also assign a contribution score to each feature and thus provide an interpretation of feature
importance at each data point.

## The code can be run as follow:

`python NN_CR.py filename regularization_scale whether_to_use_saved_train_test_lists`

The 'filename' is the dataset you want to run the program on and 'regularization_scale' is the lasso penalty parameter that
restricts the complexity of the model. For the 'whether_to_use_saved_train_test_lists' argument, if set to 0, the program
will randomly divide the data into training and testing set, if set to 1, the program will load already saved training and
testing set divisions (which are named 'train_list' and 'test_list').

For instance, the parameter settings mentioned in our paper will be the following:

`python NN_CR.py combined_74 0.0001 0`

## Dataset format
We have uploaded our dataset file as an example for running our program. In general, the first line of the file is the 
column name and the data section starts from the second line. Each line of data is formatted as follow:

|whether_the_region_contains_CircRNA           |        region_location                 |            features |
|-----------------------|----------------|-----------|
|Yes|chr19:44248923-44252204|0 0 0 0 1 ...|

## Output files:

|file|        content                 |
|-----------------------|----------------|
|mean.npy, scale.npy| mean and standard deviation of each feature|
|train_list, test_list| saved training and testing set divisions|
|result_at_epoch_#.npy| saved results at each epoch|
|checkpoint| check point file of tensorflow|
|model_#.ckpt.*| save files of tensorflow|

Inside the 'result_at_epoch_#.npy' files, the results are arranged in the following order:

|normalized features|  contribution of each feature     | confidence score ( likely hood of this data point has label 0) | model prediction of label     |  actual label    |
|-----------------------|----------------|----------------|----------------|----------------|
|0.13896221, -0.13456792...|  -9.9003075e-05, -1.1805152e-05...     | 0.72189313 | 0     |  0    |

