# DeepSynergy + XAI

## DeepSynergy:

“DeepSynergy: predicting anti-cancer drug synergy with Deep Learning” by Kristina Preuer, et al (2017) 

## Explainability- SHAP:

“A Unified Approach to Interpreting Model Predictions” by Scott M. Lundberg, et al (2017)

>Pipeline/ Workflow

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

>Dataset

23062 samples:

38 drugs- 583 distinct combinations

39 cell lines

583*39 +supplementary drug combos

- Drug A
- Drug B
- Cell line
- Synergy score

SMILES- Simplified Molecular Input Line Entry System

>Step 1: Feature vectors generation

 The feature vectors given in the base paper cannot be used as explainability needs feature vector headings which is not available in the dataset given. 

 According to DeepSynergy:

 - 1309- ECFP_6

 - 802-Physico-Chemical Descriptors

 - 2276- Toxicophore features 

 - 3984 Genomic features 

 #### Steps to generate new feature vectors:

1.	Use labels to get drug names
2.	Use SMILES values to generate the input vectors for drugs
3.	Generate ECFP using RDKit
4.	Generate standard Physico-chemical descriptors using ChemoPy and RDKit
5.	Generate Toxicophore features from rd_filters.
6.	Extract gene features from literature and perform normalization.
7.	Combine all the features according to the distinct combinations given in labels file.
8.	Reduce feature space by filtering out zero variance features.

#### New input feature vectors:

- ECFP_6 :

    - Extended connectivity finger print
    - Generated for 38 drugs using RDKit
    - 1024 bits(features) for each drug
    - 2048 ECFP per row

- Physico-Chemical Descriptors

    - Using ChemoPy
    - For each drug-632 features
    - 632*2 -1264 features per row

- Toxicophore features

    - For each row value of column is 0/1 showing the presence and absence of particular toxicophore in drug.
    - Source for SMARTS- alert_collection.csv from rd_filters public SMARTS toxicophore features
    - A toxicophore is a substance that is known to be toxic. SMARTS encode these toxic substructures and enable pattern matching. 
    - rd_filters is a public python library.
    - DeepSynergy:
            *The set of chemical features was completed by binary toxicophore features based on a set of toxicophores collected from the literature. Toxicophores are substructures which are known to be toxic (Singh et al., 2016).*

- Cell line features

     - Each cell line- top 3984 gene features
    - Normalized values
    - Source- ArrayExpress E-MATB-3610
    - Normalized using FARMS

#### Concatenation 
Concatenation to get final dataset with labelled feature vectors by matching values of the files with labels file:

- drug_a_name
- drug_b_name
- cell_line
- A_ECFP1...A_ECFP1024
- B_ECFP1...B_ECFP1024
- A_Physicochemical_descriptors...
- B_Physicochemical_descriptors...
- A_Toxicophore_features...
- B_Toxicophore_features...
- cell_line_gene_features...
- synergy_score

Saved: C:\DeepSynergy\dataset\Feature_vectors_dataset.csv

Final shape: (23052, 11072)

- 23052 rows which are the unique combinations of 38 drugs and 39 cell lines
- 11072 features per row which include ECFP, physicochemical descriptors, Toxicophore features and cell line gene features

>Step 2:Data Preprocessing

According to DeepSynergy: *The chemical feature
 space was reduced by filtering out zero variance features*

Original shape: (23052, 11072)

Filtered shape: (23052, 7958)

**Final dataset: 23052 X 7958**

>Step 3: Hyperparamter tuning

- It is neccessary to find the best hyperparamters to train the model with
- Stratified nested cross validation with 5 folds
- Using GridSearch CV
- Hyperparameters:
    
    - Preprocessing technique
    - Learning rate
    - dropout value
    - no. of hidden layers

    ![Hyper Image](C:\\Users\\mayak\\OneDrive\\Desktop\\Yazhini\\hyper.png)












