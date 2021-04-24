# Sudoku
Final Project for CSCE 5210

This project is a part of solution to the CV and Neural Network based Sudooku solver.

This has two major components
1. A neural network based digit recognizer: This part is done by creating a keras based sequential model that is created with a custom digit data.
2. A CV based image processing component that process input data to the state that model requires.

See below file(s) for:
OCR_CNN.ipynb for the model training and creation process.
myData.tar.gz file for the training data (digits 0 through 9) arranged in individual sub folders.
Sudoku.ipynb for image preprocessing and digit prediction. This file creates a 9x9 2D array with predicted numeric values.

This 2D output from above process is the input for the Sudoku Solver hosted at the below git repo:
https://github.com/Gayatri345/Sudoku
