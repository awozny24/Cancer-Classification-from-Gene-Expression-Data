# Cancer-Classification-from-Gene-Expression-Data
This project uses gene expression data, which contains many features but few samples, to predict the presence of lung cancer and specify the corresponding type, given gene expression data of a lung tissue sample.

Because there are few samples compared to the numerous features, a feature selection technique, Effective Range based Gene Selection (ERGS), will be used to select the most relevant gene for predicting the cancer in a sample.  

Different models, including SVM (using Linear, Polynomial, and Gaussian kernels), Naive Bayes, and Logistic Regression, will be used for prediction, along with a Logistic Regression meta-model created from an ensemble stacking approach built on these base classifiers.  
