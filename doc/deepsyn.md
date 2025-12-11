# DeepSynergy + XAI

## DeepSynergy:

“DeepSynergy: predicting anti-cancer drug synergy with Deep Learning” by Kristina Preuer, et al (2017) 

DeepSynergy is a neural network model created to predict synergy score for a particular drug combination applied over a cell line.

## Explainability- SHAP:

“A Unified Approach to Interpreting Model Predictions” by Scott M. Lundberg, et al (2017)

SHAP (SHapley Additive exPlanations). 

SHAP assigns each feature an importance value for a particular prediction

>**Pipeline/ Workflow**

1.	Feature vector generation
2.	Data preprocessing- perform zero variance filtering
3.	Hyper Parameter tuning- Stratified nested cross validation with 5 folds
4.  Perform Normalization
5.	Train model
6.	Apply SHAP to identify top values driving prediction- predicted important features
7.	Drug feature extraction- DrugBank, KEGG
8.	Cell line feature extraction from CCLE- gene expressions and mutations
9.	Combine features- Drug+ Cell line
10.	Intersect drug target and pathways to cell line pathways- biologically important features
11.	Overlap SHAP with Ground truth

