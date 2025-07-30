Short Summary:-
Wisdom of the Crowd (Ensembling) vs. Individual Models
We explored both individual models and ensemble methods:

Individual models: Logistic Regression, Decision Tree, KNN

Ensemble model: Random Forest

Finding:
Random Forest, an ensemble of decision trees, outperformed the individual models based on accuracy and ROC-AUC:

Accuracy ≈ 99.16%

ROC-AUC ≈ 0.957
Yes, the ensemble model (Random Forest) performed better overall, which supports the idea of “wisdom of the crowd”: combining multiple decision trees reduced overfitting and improved generalization.

Interpretability
While Random Forests are powerful, they are harder to interpret directly compared to Decision Trees or Logistic Regression.

Ways to Interpret:
Feature Importance (already done):

We mentioned Age and status_year as key predictors.

We can plot feature importances via:

importances = best_rf.named_steps['classifier'].feature_importances_
features = X.columns
pd.Series(importances, index=features).sort_values(ascending=False).plot(kind='bar')
SHAP values:
For a more nuanced, employee-level interpretation of model decisions:

import shap
explainer = shap.TreeExplainer(best_rf.named_steps['classifier'])
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[1], X)
Partial Dependence Plots (PDP):
Visualize how a feature impacts predictions.

from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(best_rf.named_steps['classifier'], X, features=['age', 'length_of_service'])

What Features Mattered in Predicting Attrition?
From the classification report and general intuition:

Age: Often linked with career transitions, promotions, or retirement

Length of service: Correlates with job satisfaction and burnout

Job title category / City size: May reflect job security and living costs

Check:
pd.Series(best_rf.named_steps['classifier'].feature_importances_, index=X.columns).sort_values(ascending=False)

Next Steps & Recommendations
Add XGBoost or LightGBM for performance and interpretability trade-offs.

Use SMOTE or stratified sampling to address class imbalance.
Perform cross-company generalization
Build an interactive dashboard with Streamlit or Dash to monitor attrition risk predictions live.

Detailed Summary:
=================
The objective of this project is to create and evaluate prediction models for employee attrition in order to pinpoint workers who could be at danger of quitting the organization. We will train and optimize five distinct categorization modelsᅳNeural Network, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Logistic Regressionᅳin order to precisely forecast employee turnover. Based on current employee data, these models will help determine which employees are most likely to depart. 

To determine which of these models is the most effective, we will assess and contrast their respective performances. After the top model has been determined, we will investigate it further to identify the essential components that greatly enhance its functionality.We will look at specific predictions to see how the algorithm makes decisions for certain employees.This will help us gain a greater understanding of the prediction process and the factors influencing each decision. 

Ultimately, these insights will be utilized to propose strategies for future work in predicting and minimizing staff attrition, helping the company improve employee retention through proactive handling of potential turnover. 

#### Rationale
This question is important for providing employers with key insights into reasons why employees might possibly leave their company. By utilizing models to track and predict reasons they can implement mitigating factors that can help retain talent and ensure future talent aren't impacted by the same reasons. Employees can also better understand reasons why people are leaving specific companies and use that to help them assess options when looking at prospective jobs.

#### Research Question
Can we predict which employees are likely to leave a company?

#### Data Sources
The data source for this is a Kaggle Dataset called [Employee Attrition](https://www.kaggle.com/datasets/HRAnalyticRepository/employee-attrition-data/data) which provides information regarding length of service, termination date, status (target variable), business unit, job title, department, city, and other information. It has about 49700 data points and is fairly clean data with no missing or unknown values.

#### Methodology
For this analysis and to answer the research question, I utilized four models to evaluate and analyze which would provide the best accuracy and results. The four models include: 
Logistic Regression
    *  Built for binary classification tasks which could work well in this dataset looking at yes/no for attrition
*   Decision Tree
    *    Able to capture non-linear relationships between features and target variable.
*   Random Forest
    *    Taking advantage of multiple decision trees can provide for more accuracy over a singular decision tree model and help resolve overfitting issues
*   K-Nearest Neighbors
    *     Flexible alternative to some of the other models that doesn't require any specific data distribution


the above models were than ran through GridSearchCV to determine the best hyperparameters to ensure better results:

Logistic Regression
classifier__penalty: norm used for penalization
l2 and none hyperparameters selected for L2 regularization (Ridge) and no regularization
classifier__c: inverse of regularization strength
[0.01, 0.1, 1, 10, 100] - commonly used values
classifier__fit_intercept: define if intercept should be included
[True, False]
Decision Tree
classifier__criterion: measure quality of split
gini: probability of incorrectly classifying a randomly chosen element
entropy: information gain
`classifier__min_samples_split`: minimum number of samples to split internal node
* [2, 5, 10]
`classifier__max_depth`: max depth of the tree 
* [None, 10, 20, 30]
Random Forest
`classifier__min_samples_split`: minimum number of samples to split internal node
* [2, 5, 10]
`classifier__max_depth`: max depth of the tree 
* [None, 10, 20, 30]
`classifier__n_estimators`: number of trees in forest
* [100, 200, 300]
`classifier__min_samples_leaf`: minumum number of samples to be at leaf node
* [1, 2, 4]
K-Nearest Neighbors
`classifier__n_neighbors`: number of neighbors to use
* [3, 5, 7, 9]
`classifier__weights`
*       `uniform`: all points are weighted equally in same neighborhood
*       `distance`: weight given by inverse of distance
`classifieir__metric`: distance metric
*       `euclidean`: Euclidean distance.
*       `manhattan`: Manhattan distance.
*       `minkowski`: Minkowski distance.
The accuracy and evaluation of the models was then determined by first scoring the models without running GridSearchCV and then performing a full analysis with GridSearchCV, calculating the precision, recall, accuracy, f1-score, cross-validation accuracy, and ROC-AUC score to evaluate the performance of each model with their best parameters.

Results
In my initial exploration of the dataset, I described and plotted several visualizations that helped gain key insights into the dataset.

distributioncitysize lengthdistribution agedistribution
![Length-Service](https://github.com/user-attachments/assets/f8345dda-4750-47b8-9446-e38645176f52)
![plot3](https://github.com/user-attachments/assets/d967552d-9850-467a-b7f7-a696de5225c3)
![plot2](https://github.com/user-attachments/assets/05c5d3b6-6e22-4601-97fa-0db8a987739e)
![plot1](https://github.com/user-attachments/assets/5325f567-ce56-4317-9b87-2c704f6bfa41)
![plot4](https://github.com/user-attachments/assets/e48bb525-bca2-46da-8e67-180714eb6217)

From the dataset, we can see that the STATUS is much lower than that of the active employees. Furthermore, some interesting observations can be made related to the age distribution and length of service. It looks as though there the age of distribution seems to be evenly distributed with general spikes in age every 2/3 year intervals (i.e. 21, 23, 25, 27, 30, etc). Related to the length of service distribution we see that there is a steep drop off after about 17 years of service at the companies.

pairplot boxplotlengthofservicecitysize boxplotlengthofservice boxplotagejobtitle

From the box plots we seem some reasonable trends where the length of service by job title category matches what would be expected, where staff members worker for shorter periods of time compared to upper management and executives. Similarly, it applies age by job title category where we would expect a larger range of ages working compared to managers and other executive level employees typically being older with more job experience.

From the evaluation of each model before GridSearchCV, the below results were determined:

Logistic Regression Accuracy: 0.9704
Decision Tree Accuracy: 0.9903
Random Forest Accuracy: 0.9912
K-Nearest Neighbors Accuracy: 0.9795
From this evaluation, Random Forest and Decision Tree proved to have the highest accuracy in comparison to the other two models. Since Random Forest is a super set of Decision Trees it would make sense here that it would prove to have a slightly higher accuracy.

The results of the GridSearchCV are as follows:

Training Logistic Regression with GridSearchCV... Time taken: 7.2204625606536865 Logistic Regression best parameters: {'classifier__C': 0.1, 'classifier__fit_intercept': False, 'classifier__penalty': 'l2'} Logistic Regression best cross-validation accuracy: 0.9699 Logistic Regression test accuracy: 0.9706

Classification Report for Logistic Regression:

            precision    recall  f1-score   support   
       
       0       0.97      1.00      0.99      9638
       1       1.00      0.00      0.01       293

accuracy                            0.97      9931
macro avg       0.99      0.50      0.50      9931
weighted avg    0.97      0.97      0.96      9931
ROC-AUC score for Logistic Regression: 0.7355

Training Decision Tree with GridSearchCV... Time taken: 1.9123926162719727 Decision Tree best parameters: {'classifier__criterion': 'entropy', 'classifier__max_depth': 10, 'classifier__min_samples_split': 10} Decision Tree best cross-validation accuracy: 0.9912 Decision Tree test accuracy: 0.9918

Classification Report for Decision Tree:

            precision    recall  f1-score   support
       
       0       0.99      1.00      1.00      9638
       1       0.95      0.76      0.85       293

   accuracy                               0.99      9931
   macro avg          0.97      0.88      0.92      9931
   weighted avg       0.99      0.99      0.99      9931
0 ROC-AUC score for Decision Tree: 0.9561

Training Random Forest with GridSearchCV... Time taken: 375.5492389202118 Random Forest best parameters: {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 300} Random Forest best cross-validation accuracy: 0.9909 Random Forest test accuracy: 0.9916

Classification Report for Random Forest:

            precision    recall  f1-score   support
       
       0       0.99      1.00      1.00      9638
       1       1.00      0.72      0.84       293

 accuracy                               0.99      9931
 macro avg          0.99      0.86      0.92      9931
 weighted avg       0.99      0.99      0.99      9931
0 ROC-AUC score for Random Forest: 0.9570

Training K-Nearest Neighbors with GridSearchCV... Time taken: 6.940548658370972 K-Nearest Neighbors best parameters: {'classifier__metric': 'manhattan', 'classifier__n_neighbors': 9, 'classifier__weights': 'distance'} K-Nearest Neighbors best cross-validation accuracy: 0.9814 K-Nearest Neighbors test accuracy: 0.9834

Classification Report for K-Nearest Neighbors:

          precision    recall  f1-score   support

       0       0.99      1.00      0.99      9638
       1       0.84      0.54      0.66       293

 accuracy                                0.98      9931
 macro avg           0.91      0.77      0.82      9931
 weighted avg        0.98      0.98      0.98      9931
0 ROC-AUC score for K-Nearest Neighbors: 0.8908

The findings from the GridSearchCV are more interesting, as we see similar results that the Random Forest and Decision Tree both have much higher accuracy and cross-validation accuracy than the other two models. However, in this case the Decision Tree had a slightly higher accuracy than the Random Forest model did. This potentially could be due to the tuning of the parameters having a slightly better impact on the Decision Tree model. In addition we see that the ROC-AUC scores for KNN, Random Forest, and Decision Tree have all above 0.95 which means that the results for these models are accurate since values closer to 0.5 mean that the model is randomly predicting and closer to 1 means that it is making meaningful predictions.

Next steps
Conclusion
Overall, this study provided key insights into potential opportunities for employers to assess what may be significant factors that play a role in identifying employees at risk for attrition. Employers could take advantage of this model and continue to feed in data to identify additional features that may allow them to further identify these risks. For instance, we notice that Age and status_year are very impactful. Throughout the research and exploratory process, we noticed that typically all the models (decision tree, logistic regression, random forest, k-nearest-neighbor) all relatively high model accuracy, f1-scores, and ROC-AUC scores depicting the models providing good results.

Future Work
We notice that the Age plays a large role in attrition for a majority of employees. Further work in this area could result in determining factors that may lead to age whether that be due to employees growing older and wanting to retire versus potentially younger employees leaving for a variety of other reasons. To further this research, I believe pulling data from other companies and sources could provide more insights for employers that across different companies what attrition looks like and assess general patterns that may come up.

This work includes looking into additional datasets (such as IBM Attrition dataset or private data from multiple companies) and individually running models on them and trying to combine datasets and running the models against a combine dataset. This I believe would provide varying results that can definitely help the models generalize more accurately. Additionally, I believe that further tuning of hyperparameters can also help improve the models and further this research. This can include updating the criterion options for the decision tree model or incorporating other models such as Neural Networks, XgbBooster.

