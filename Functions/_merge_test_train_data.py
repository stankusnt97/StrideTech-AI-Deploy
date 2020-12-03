def merge_test_train(test_initial_cleaned_data):
        with open('/Users/stankusnt/Desktop/Work/StrideTech AI Test/Model_training/initial_data_cleaned.csv', 'a') as f:
            (test_initial_cleaned_data).to_csv(f, header=False, index=False)