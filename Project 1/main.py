import project1 as p1
import utils

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)
#
#-------------------------------------------------------------------------------
# Section 1.7
#-------------------------------------------------------------------------------
#toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')
#
#T = 10
#L = 0.2
#
#thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
#thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
#thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)
#
#def plot_toy_results(algo_name, thetas):
#    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)
#
#plot_toy_results('Perceptron', thetas_perceptron)
#plot_toy_results('Average Perceptron', thetas_avg_perceptron)
#plot_toy_results('Pegasos', thetas_pegasos)
#-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.9.b
#-------------------------------------------------------------------------------
#T = 10
#L = 0.01
#
#pct_train_accuracy, pct_val_accuracy = \
#   p1.perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
#print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
#print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))
#
#avg_pct_train_accuracy, avg_pct_val_accuracy = \
#   p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
#print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
#print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))
#
#avg_peg_train_accuracy, avg_peg_val_accuracy = \
#   p1.pegasos_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
#print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
#print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))
##-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.10
#-------------------------------------------------------------------------------
#data = (train_bow_features, train_labels, val_bow_features, val_labels)
#
## values of T and lambda to try
#Ts = [1, 5, 10, 15, 25, 50, 100]
#Ls = [0.01, 0.1, 0.2, 0.5, 1]
#
#pct_tune_results = utils.tune_perceptron(Ts, *data)
#avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
#
## fix values for L and T while tuning Pegasos T and L, respective
#best_L = 0.01
#best_T = 95
#
#avg_peg_tune_results_T = utils.tune_pegasos_T(best_L, Ts, *data)
#avg_peg_tune_results_L = utils.tune_pegasos_L(best_T, Ls, *data)
#
#utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
#utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
#utils.plot_tune_results('Pegasos', 'T', Ts, *avg_peg_tune_results_T)
#utils.plot_tune_results('Pegasos', 'L', Ls, *avg_peg_tune_results_L)
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11a
#
# Call one of the accuracy functions that you wrote in part 2.9.a and report
# the hyperparameter and accuracy of your best classifier on the test data.
# The test data has been provided as test_bow_features and test_labels.
#-------------------------------------------------------------------------------
#print(p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=15))
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11b
#
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
#best_theta = p1.average_perceptron(train_bow_features, train_labels, T=15)[0]
#wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
#sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
#print("Most Explanatory Word Features")
#print(sorted_word_features[:10])
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.12
#
# After constructing a final feature representation, use code similar to that in
# sections 2.9b and 2.10 to assess its performance on the validation set.
# You may use your best classifier from before as a baseline.
# When you are satisfied with your features, evaluate their accuracy on the test
# set using the same procedure as in section 2.11a.
#-------------------------------------------------------------------------------

## Base Case
dictionary = p1.bag_of_words(train_texts)

train_final_features = p1.extract_final_features(train_texts, dictionary)
val_final_features   = p1.extract_final_features(val_texts, dictionary)
test_final_features  = p1.extract_final_features(test_texts, dictionary)

print(p1.average_perceptron_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=15))
print(p1.average_perceptron_accuracy(val_final_features,test_final_features,val_labels,test_labels,T=15))

## Improve 1
dictionary = p1.bag_of_better_words(train_texts)

train_final_features = p1.extract_final_features(train_texts, dictionary)
val_final_features   = p1.extract_final_features(val_texts, dictionary)
test_final_features  = p1.extract_final_features(test_texts, dictionary)

#data = (train_final_features, train_labels, val_final_features, val_labels)
#
## values of T and lambda to try
#Ts = [1, 5, 10, 15, 25, 50, 100]
#
#avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
#utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)

print(p1.average_perceptron_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=15))
print(p1.average_perceptron_accuracy(val_final_features,test_final_features,val_labels,test_labels,T=15))

## Improve 2
theta = p1.average_perceptron(train_bow_features, train_labels, T=15)[0]
new_dictionary = p1.bag_of_elite_words(dictionary, theta, 0.25)

train_final_features = p1.extract_final_features(train_texts, new_dictionary)
val_final_features   = p1.extract_final_features(val_texts, new_dictionary)
test_final_features  = p1.extract_final_features(test_texts, new_dictionary)

#data = (train_final_features, train_labels, val_final_features, val_labels)
#
## values of T and lambda to try
#Ts = [1, 5, 10, 15, 25, 50, 100]
#
#avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
#utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)

print(p1.average_perceptron_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=100))
print(p1.average_perceptron_accuracy(val_final_features,test_final_features,val_labels,test_labels,T=100))

## Improve 2b
theta = p1.average_perceptron(train_bow_features, train_labels, T=15)[0]
new_dictionary = p1.bag_of_elite_words(dictionary, theta, 0.20)

train_final_features = p1.extract_final_features(train_texts, new_dictionary)
val_final_features   = p1.extract_final_features(val_texts, new_dictionary)
test_final_features  = p1.extract_final_features(test_texts, new_dictionary)
#
##data = (train_final_features, train_labels, val_final_features, val_labels)
##
### values of T and lambda to try
##Ts = [1, 5, 10, 15, 25, 50, 100]
##
##avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
##utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
#
print(p1.average_perceptron_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=100))
print(p1.average_perceptron_accuracy(val_final_features,test_final_features,val_labels,test_labels,T=100))

#
theta = p1.average_perceptron(train_bow_features, train_labels, T=15)[0]
new_dictionary = p1.bag_of_elite_words(dictionary, theta, 0.70)

train_final_features = p1.extract_final_features(train_texts, new_dictionary)
val_final_features   = p1.extract_final_features(val_texts, new_dictionary)
test_final_features  = p1.extract_final_features(test_texts, new_dictionary)

print(p1.average_perceptron_accuracy(train_final_features,val_final_features,train_labels,val_labels,T=100))
print(p1.average_perceptron_accuracy(val_final_features,test_final_features,val_labels,test_labels,T=100))


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.13
#
# Modify the code below to extract your best features from the submission data
# and then classify it using your most accurate classifier.
#-------------------------------------------------------------------------------
# submit_texts = [sample['text'] for sample in utils.load_data('reviews_submit.tsv')]
#
# # 1. Extract your preferred features from the train and submit data
# dictionary = p1.bag_of_words(submit_texts)
# train_final_features = p1.extract_final_features(train_texts, dictionary)
# submit_final_features = p1.extract_final_features(submit_texts, dictionary)
#
# # 2. Train your most accurate classifier
# final_thetas = p1.perceptron(train_final_features, train_labels, T=1)
#
# # 3. Classify and write out the submit predictions.
# submit_predictions = p1.classify(submit_final_features, *final_thetas)
# utils.write_predictions('reviews_submit.tsv', submit_predictions)
#-------------------------------------------------------------------------------
