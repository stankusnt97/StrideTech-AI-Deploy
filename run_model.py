from Functions._main import model_test

error, predictions, predictions_and_labels = model_test(Name='Nick Stankus', standing_filepath='/Users/stankusnt/Desktop/StrideTech/ML Data/standing_nick_stankus_test_data.txt', sitting_filepath='/Users/stankusnt/Desktop/StrideTech/ML Data/sitting_nick_stankus_test_data.txt', walking_filepath='/Users/stankusnt/Desktop/StrideTech/ML Data/walking_nick_stankus_test_data.txt', add_to_train_data=True)

print(predictions_and_labels)