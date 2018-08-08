# machine-learning-with-pytorch

#### Summary of Project

The project creates an artificial neural network using PyTorch and trains it to identify articles of clothing. The clothing data is stored at this URL:
http://fashionmnist.s3-website.eu-central-1.amazonaws.com. I was unable to upload the data folder to Github, so when you run the program for the first time it will download
the data automatically from the web and store it in a folder called 'data' in the current directory. This will take a while - probably close to 20 minutes. If you run the program again,
the 'data' folder will already be there and the program will not have to download it again.

The program creates a training set, a validation set, and a test set and uses 3 different neural networks to learn the data:

1. a regular network with 1 hidden layer
2. same as 1, but with an added dropout layer that potentially zeroes out the outputs of each layer 
3. same as 1, but with added batch normalization on the outputs of each layer.

The program outputs its results from training, validating, and testing on each of the three layers. The output is formatted and labeled and should be self explanatory.

#### How to Run

Assuming you have python, just open a terminal, cd to the directory containing the code file and the data folder, and enter `python ml_code.py`. See above regarding the 'data' folder.
The program downloads this folder automatically.