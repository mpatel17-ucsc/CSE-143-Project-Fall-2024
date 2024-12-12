CSE 143 Project Fall 2024 - Albert, Tanishq, and Manav

For data files, please go to https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data and extract the data file. The train.csv and test.csv should be located at the same directory as the .ipynb files.
For the GloVe file for embedding, go to https://nlp.stanford.edu/projects/glove/ and extract the data file called "glove.6B.zip". The folder called "glove.6b" should be in the same directory as the .ipynb files.

To run the LSTM + GloVe file, simply go to the file and run the cell. It will start training and printing out the accuracy of each epoch. You will see 2 accuracies: Main Accuracy and Auxiliary Accuracy. Main Accuracy is just general accuracy of whether a comment is toxic without taking account of identities, while Auxiliary Accuracy is accuracy of whether a comment is toxic with respect to minimizing bias from certain identities.