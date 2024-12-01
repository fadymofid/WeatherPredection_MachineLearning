The Fifth Assessment Report from the Intergovernmental Panel on Climate
Change (IPCC) confirms that our climate and its extreme events are changing.
To reduce the risks and damage caused by these weather and climate

extremes, accurate predictions are essential for both short-term and long-
term planning. Understanding, modeling, and predicting these extremes is a

key area of climate research, and it has been identified as one of the World
Climate Research Program's (WCRP) Grand Challenges, known as the
Extremes Grand Challenge. Although weather forecasting presents many
challenges, we have created a simple task as an introduction to this area.
Using our dataset, we aim to predict whether it will rain.
Dataset:
Our dataset is a modified version of a weather forecast dataset obtained from
Kaggle. It includes 6 features and 2,500 observations. Your task is to analyze
and work with this data to answer the following questions, using Python
code:

Requirements:
Task 1: Preprocessing
1. Does the dataset contain any missing data? Identify them.
2. Apply the two techniques to handle missing data, dropping missing
values and replacing them with the average of the feature.
3. Does our data have the same scale? If not, you should apply feature
scaling on them.
4. Splitting our data to training and testing for training and evaluating our
models

Task 2: Implement Decision Tree, k-Nearest Neighbors (kNN) and naïve
Bayes
1. Using scikit-learn implement Decision Tree, kNN and Naïve Bayes
2. Compare the performance of your implementations by evaluating
accuracy, precision, and recall metrics.
3. Implement k-Nearest Neighbors (kNN) algorithm from scratch.

4. Report the results and compare the performance of your custom k-
Nearest Neighbors (kNN) implementation with the pre-built kNN

algorithms in scikit-learn, using the evaluation metrics mentioned in
point 2. Using any missing handling techniques, you chose from task 1.2.
Task 3: Interpreting the Decision Tree and Evaluation Metrics Report
1. The effect of different data handling

o Provide a detailed report evaluating the performance of scikit-
learn implementations of the Decision Tree, k-Nearest Neighbors

(kNN) and naïve Bayes with respect to the different handling
missing data technique.
2. Decision Tree Explanation Report
o Create a well-formatted report that includes a plot of the
decision tree and a detailed explanation of how the tree makes
predictions.
o Discuss the criteria and splitting logic used at each node of the
tree.

3. Performance Metrics Report
o Provide a detailed report evaluating the performance of your
implementations of the k-Nearest Neighbors (kNN) from scratch
with different k values at least 5 values.
o Include the accuracy, precision, and recall metrics for models.
o Compare these results with the performance of the
corresponding algorithms implemented using scikit-learn.
