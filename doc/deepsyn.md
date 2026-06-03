# DeepSynergy + XAI

## DeepSynergy:

“DeepSynergy: predicting anti-cancer drug synergy with Deep Learning” by Kristina Preuer, et al (2017) 

DeepSynergy is a neural network model created to predict synergy score for a particular drug combination applied over a cell line.

## Explainability- SHAP:

“A Unified Approach to Interpreting Model Predictions” by Scott M. Lundberg, et al (2017)

SHAP (SHapley Additive exPlanations). 

SHAP assigns each feature an importance value for a particular prediction

>**Pipeline/ Workflow**

1.	Feature vectors extraction
2.	Feature vector concatenation
3.	Hyper Parameter tuning- Stratified nested cross validation with 5 folds
4.  Perform Normalization
5.	Train model
6.	Apply SHAP to identify top values driving prediction
7.	Demonstrate explainability analysis using suitable USECASE


