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

Age: Often linked with career transitions, promotions, or retirement.

Length of service: Correlates with job satisfaction and burnout.

Job title category / City size: May reflect job security and living costs.

Check:
pd.Series(best_rf.named_steps['classifier'].feature_importances_, index=X.columns).sort_values(ascending=False)

Next Steps & Recommendations
Add XGBoost or LightGBM for performance and interpretability trade-offs.

Use SMOTE or stratified sampling to address class imbalance.

Perform cross-company generalization

Build an interactive dashboard with Streamlit or Dash to monitor attrition risk predictions live.

Note: Will add more in depth details in the Readme file at the last module
