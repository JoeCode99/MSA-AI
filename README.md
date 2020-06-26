# MSA-AI
The complete AI Project for MSA.

## Table of Contents
* [Project Description](#project-description)
* [Environment Setup and Dependencies](#environment-setup-and-dependencies)
* [How to Train and Test the Model](#how-to-train-and-test-the-model)
* [Potential Improvements to the Project](#potential-improvements-to-the-project)

## Project Description
The overall goal of this project is to develop a model that is capable of classifying flower types according to the attributes provided in the ‘Iris.csv’ dataset. The dataset is incredibly simple, consisting of 5 columns and 150 rows – with only three different flower classifications to distinguish between (Iris Setosa, Iris Versicolor, and Iris Virginica). The small dataset allowed all development energy to be focused on experimenting with different supervised machine learning algorithms, evaluating their accuracies, and selecting the most accurate model to make predictions. More complicated datasets would require redundant columns to be removed from training and testing data, and may even require ‘dummy’ columns to be produced. Unsupervised ML algorithms (or deep-learning algorithms) would also considerably complicate development efforts. Given the limited time frame to work on this project (due to exams), the Iris Dataset was a perfect introduction into the machine learning world, allowing me to quickly understand a wide variety of useful concepts.

Throughout development, two factors were prioritized above anything else:
*	The model must be as accurate as possible, while avoiding the perils of overfitting and underfitting. 
*	Training practices must not introduce any sources of bias to the model. 

Both of these goals were successfully met. Evidence for this will be provided in the upcoming sections.

## Environment Setup and Dependencies
All development was done within Microsoft Azure Notebooks, writing all code into a Jupyter notebook (using Python 3.6). This proved to be extremely helpful, as very little time was dedicated to setting up the development environment – with only the notebook file needing to be created.  

Scikit-learn was by far the most influential dependency used throughout the project. The complete list of dependencies utilized for the completion of this project includes:
*	pandas.read_csv: Enabled the program to read in the ‘Iris.csv’ dataset.
*	sklearn.model_selection.train_test_split: Allowed splitting of the data into training and testing datasets. Applies random shuffling techniques under the hood.
*	sklearn.svm.SVC: Import framework for a Support Vector Machine algorithm.
*	sklearn.linear_model.LogisticRegression: Import framework for a Logistic Regression Classification algorithm.
*	sklearn.tree.DecisionTreeClassifier: Import framework for a Decision Tree Classification algorithm.
*	sklearn.model_selection.cross_val_score & sklearn.model_selection.StratifiedKFold: Allows for simple implementation of the cross-validation technique to evaluate accuracy of chosen models.
*	sklearn.metrics.accuracy_score: Calculates accuracy of model’s predictions.
*	sklearn.metrics.confusion_matrix: Produces the confusion matrix for the model’s predictions.
*	sklearn.metrics.classification_report: Produces a detailed classification report (with precision, recall, and f1-score values) for the model’s predictions.

## How to Train and Test the Model
By evaluating several different machine learning algorithms, it was found the Support Vector Classification algorithm worked best in this scenario, achieving an estimated accuracy of 98% by applying the cross validation technique (more details about this technique are provided below). To train and test the data:
1.	Perform an 80/20 split on the dataset using the train_test_split method (provided by sklearn.model_selection) to create the training and testing datasets respectively.
2.	Create the SVC model by producing an instance of the sklearn.svm.SVC object in the program.
3.	Fit the SVM to the training dataset by using the sklearn.svm.SVC.fit method. This method handles all the training logic under the hood in a single line of code.
4.	Make predictions with the trained model by calling the ‘predict’ method on the model, passing in the test dataset.
5.	Get the percentage of correct predictions by using the sklearn.metrics.accuracy_score method, passing in the correct answers as well as the predictions.
6.	Further analyze the accuracy of the predictions by calling the sklearn.metrics.confusion_matrix method, which will produce a confusion matrix of the results.

Much of the lower-level logic is automatically handled by the functions provided by scikit-learn. This is emphasized by the program’s use of cross validation to determine the best possible ML algorithm/model for the dataset, requiring only two lines of code to be completed. Under the hood, the following steps are conducted for cross validation:
1.	Shuffle the data randomly.
2.	Split the data into groups (10 in this case).
*	3. For each group:
    3.1 Pick one group as the test dataset and treat the rest as the training datasets.
    3.2 Apply the model to the training datasets and evaluate its accuracy on the test set.
4.	Return the mean of the evaluation scores to determine the model’s performance.

## Potential Improvements to the Project
The program (as seen now) only evaluates the effectiveness of three different supervised ML algorithms. A wider variety of ML algorithms could have been tested all at once without adding much complexity (scikit-learn provides plenty of libraries to make this incredibly straightforward). Other algorithms were tested (such as the Gaussian Naïve Bayes algorithm) but I decided to reduce the final program to only display comparisons between the top three contenders.

With more time I could have tackled a more complex dataset - such as the Parkinson’s dataset -, which would have required me to apply other ML techniques such as removing redundant fields from analysis. Given the open-source nature of these datasets, I will definitely work on these interesting datasets at a later date.
