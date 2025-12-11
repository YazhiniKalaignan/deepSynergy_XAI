# Hyperparamter tuning

>Step 3: Hyper Parameter tuning

According to DeepSynergy, before hyperparameter tuning, we perform normalization using 3 techniques:

- norm (Z-score normalization)
- tanh (Tanh scaling)
- tanh_norm (Tanh followed by Z-score normalization)

After normalization: 

- It is neccessary to find the best hyperparamters to train the model with
- Stratified nested cross validation with 5 folds
- Using GridSearch CV
- Hyperparameters:
    
    - Preprocessing technique (norm; norm+tanh; norm+tanh+norm)
    - Learning rates ($10^2$;$10^3$;$10^4$;$10^5$)
    - dropout values (no dropout; input: 0.2, hidden: 0.5)
    - no. of hidden layers ( [8192, 8192]; [4096, 4096]; [2048, 2048];
[8192, 4096]; [4096, 2048]; [4096, 4096, 4096];
[2048, 2048, 2048]; [4096, 2048, 1024];
[8192, 4096, 2048] )
