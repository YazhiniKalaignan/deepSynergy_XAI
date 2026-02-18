# Model building

>Step 3: Hyper Parameter tuning

According to DeepSynergy, we perform normalization using 3 techniques:

- norm (Z-score normalization)
- tanh (Tanh scaling)
- tanh_norm (Tanh followed by Z-score normalization)

- It is neccessary to find the best hyperparamters to train the model with
- Stratified nested cross validation with 5 folds
- Using GridSearch CV
- Best hyperparamter- least validation error
- Hyperparameters:
    
    - Normalization technique (norm; norm+tanh; norm+tanh+norm)
    - Learning rates ($10^2$ ; $10^3$ ; $10^4$ ; $10^5$)
    - dropout values (no dropout; input: 0.2, hidden: 0.5)
    - no. of neurons in the hidden layers  ([4096, 2048], [2048, 1024], [4096, 2048, 1024], [2048, 1024, 512], [1024, 1024])


- Best hyperparameter configuration:
       
        Layers (neurons) = [1024,1024]

        Hidden_Dropout= 0.5

        Input_dropout= 0.2

        Learning rate = 0.0001

        Normalization technique= 'norm'
    
- Model metrics:

    Validation Metrics:
        -Mean absolute error               :0.4545
        -Mean squared error                :0.5046
        -Root mean squared error           :0.7104
        -Pearson's correlation coefficient :0.9995

    
    
