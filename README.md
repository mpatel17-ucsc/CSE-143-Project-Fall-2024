CSE 143 Project Fall 2024 - Albert, Tanishq, and Manav

For data files, please go to https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data and extract the data file. The train.csv and test.csv should be located at the same directory as the .ipynb files.
For the GloVe file for embedding, go to https://nlp.stanford.edu/projects/glove/ and extract the data file called "glove.6B.zip". The folder called "glove.6b" should be in the same directory as the .ipynb files.

To run the LSTM + GloVe file, simply go to the file and run the cell. It will start training and printing out the accuracy of each epoch. You will see 2 accuracies: Main Accuracy and Auxiliary Accuracy. Main Accuracy is just general accuracy of whether a comment is toxic without taking account of identities, while Auxiliary Accuracy is accuracy of whether a comment is toxic with respect to minimizing bias from certain identities.

To run the BERT + POS tagging file, simply go to the file, install the necessary dependencies, and run the cell. This is memory-intensive, so the default is set to not train the BERT model directly. You will see the model train itself and create a submission.csv containing the model's predictions. During training, we track the ROC-AUC on the binary classification task so as to allow you to get an idea of what the model is predicting. 