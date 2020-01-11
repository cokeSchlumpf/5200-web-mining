import os

os.system('echo Pearson Coefficient')
os.system('echo -------------------')
os.system('dvc metrics show -a -T --type json --xpath validation.PearsonCorrelation')
os.system('echo')

os.system('echo Mean Absolute Error')
os.system('echo -------------------')
os.system('dvc metrics show -a -T --type json --xpath validation.MeanAbsoluteError')
os.system('echo')

os.system('echo Mean Squared Error')
os.system('echo -------------------')
os.system('dvc metrics show -a -T --type json --xpath validation.MeanSquaredError')
os.system('echo')

os.system('echo Accuracy, Recall, F1 Score')
os.system('echo --------------------------')
os.system('dvc metrics show -a -T --type json --xpath validation.Accuracy')
os.system('echo')
