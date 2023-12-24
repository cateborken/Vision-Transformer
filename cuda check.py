# Extract train and test accuracies from the progress
train_accuracies = [0.4335, 0.5744, 0.6196, 0.6302, 0.6626, 0.6715, 0.6933, 0.7028, 0.7028, 0.7143, 0.7479, 0.7367, 0.7427, 0.7551, 0.7570, 0.7770, 0.7706, 0.7874, 0.7715, 0.7777, 0.7921, 0.7902, 0.7743, 0.7915, 0.8010]
test_accuracies = [0.6074, 0.6457, 0.6615, 0.6710, 0.6769, 0.6771, 0.6802, 0.6976, 0.7019, 0.6926, 0.6992, 0.6974, 0.6927, 0.6943, 0.6976, 0.7097, 0.7019, 0.7035, 0.6988, 0.7113, 0.7082, 0.7082, 0.7144, 0.7052, 0.7082]

# Calculate overall accuracy for train and test without numpy
overall_train_accuracy = sum(train_accuracies) / len(train_accuracies)
overall_test_accuracy = sum(test_accuracies) / len(test_accuracies)

# Print the results
print(f"Overall Train Accuracy: {overall_train_accuracy:.4f}")
print(f"Overall Test Accuracy: {overall_test_accuracy:.4f}")
