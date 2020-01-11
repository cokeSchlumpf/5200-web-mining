import os

print("Pearson Coefficient")
print("-------------------")
os.system('dvc metrics show -a -T --type json --xpath validation.PearsonCorrelation')
print()

print("Mean Absolute Error")
print("-------------------")
os.system('dvc metrics show -a -T --type json --xpath validation.MeanAbsoluteError')
print()

print("Accuracy, Recall, F1 Score")
print("--------------------------")
os.system('dvc metrics show -a -T --type json --xpath validation.Accuracy')
print()
