# Restricted-Boltzmann-Machine
Collaborative Filtering is a method used by recommender systems to make predictions about an interest of an specific user by collecting taste or preferences information from many other users. The technique of Collaborative Filtering has the underlying assumption that if a user A has the same taste or opinion on an issue as the person B, A is more likely to have B’s opinion on a different issue. 

In this project I use a Restricted Boltzmann to predict whether a user would like a movie or not based on this user's taste and the taste of other users who watched and rated the same and similar movies.

# Prerequisites
- Python 3.6
- TensorFlow 1.5 or higher
- NumPy 1.11 or higher

# Datasets

The current version support only the MovieLens ml-1m.zip dataset obtained from https://grouplens.org/datasets/movielens/.

# How to Use

- Download the ml-1m.zip dataset from https://grouplens.org/datasets/movielens/.

- Save ratings.dat under DATA_DIR. Devide the ratings.dat into training and testing datasets train.dat and test.dat with the shell command:
` python train_test_split.py DATA_DIR OUTPUT_DIR_TRAIN OUTPUT_DIR_TEST`

- Use shell to make TF_Record files out of the both train.dat and test.dat files by executing the command:
`python tf_record_writer.py OUTPUT_DIR_TF_TRAIN OUTPUT_DIR_TF_TEST`

- Use shell to start the training by executing the command (optionally parse your hyperparameters):

        python train.py \
             --tf_records_train_path=OUTPUT_DIR_TF_TRAIN \
             --tf_records_test_path=OUTPUT_DIR_TF_TEST \

    epoch_nr: 0, batch: 50/188, acc_train: 0.721, acc_test: 0.709
    epoch_nr: 0, batch: 100/188, acc_train: 0.744, acc_test: 0.704
    epoch_nr: 0, batch: 150/188, acc_train: 0.748, acc_test: 0.736
    epoch_nr: 1, batch: 50/188, acc_train: 0.767, acc_test: 0.764
    epoch_nr: 1, batch: 100/188, acc_train: 0.758, acc_test: 0.733
    epoch_nr: 1, batch: 150/188, acc_train: 0.756, acc_test: 0.732
    epoch_nr: 2, batch: 50/188, acc_train: 0.772, acc_test: 0.773
    epoch_nr: 2, batch: 100/188, acc_train: 0.758, acc_test: 0.769
    epoch_nr: 2, batch: 150/188, acc_train: 0.754, acc_test: 0.771
    epoch_nr: 3, batch: 50/188, acc_train: 0.767, acc_test: 0.725
    epoch_nr: 3, batch: 100/188, acc_train: 0.758, acc_test: 0.757
    epoch_nr: 3, batch: 150/188, acc_train: 0.756, acc_test: 0.760
    epoch_nr: 4, batch: 50/188, acc_train: 0.768, acc_test: 0.717
    epoch_nr: 4, batch: 100/188, acc_train: 0.756, acc_test: 0.743
    epoch_nr: 4, batch: 150/188, acc_train: 0.759, acc_test: 0.781
    epoch_nr: 5, batch: 50/188, acc_train: 0.772, acc_test: 0.769
    epoch_nr: 5, batch: 100/188, acc_train: 0.762, acc_test: 0.774
    epoch_nr: 5, batch: 150/188, acc_train: 0.760, acc_test: 0.775
    epoch_nr: 6, batch: 50/188, acc_train: 0.774, acc_test: 0.771
    epoch_nr: 6, batch: 100/188, acc_train: 0.764, acc_test: 0.776
    epoch_nr: 6, batch: 150/188, acc_train: 0.765, acc_test: 0.775
    epoch_nr: 7, batch: 50/188, acc_train: 0.779, acc_test: 0.780
    epoch_nr: 7, batch: 100/188, acc_train: 0.765, acc_test: 0.778
    epoch_nr: 7, batch: 150/188, acc_train: 0.766, acc_test: 0.775
    epoch_nr: 8, batch: 50/188, acc_train: 0.777, acc_test: 0.785
    epoch_nr: 8, batch: 100/188, acc_train: 0.768, acc_test: 0.785
    epoch_nr: 8, batch: 150/188, acc_train: 0.764, acc_test: 0.766
    epoch_nr: 9, batch: 50/188, acc_train: 0.779, acc_test: 0.787
    epoch_nr: 9, batch: 100/188, acc_train: 0.763, acc_test: 0.784
    epoch_nr: 9, batch: 150/188, acc_train: 0.765, acc_test: 0.783
    epoch_nr: 10, batch: 50/188, acc_train: 0.775, acc_test: 0.763
    epoch_nr: 10, batch: 100/188, acc_train: 0.766, acc_test: 0.750
    epoch_nr: 10, batch: 150/188, acc_train: 0.771, acc_test: 0.784
    epoch_nr: 11, batch: 50/188, acc_train: 0.776, acc_test: 0.768
    epoch_nr: 11, batch: 100/188, acc_train: 0.765, acc_test: 0.783
    epoch_nr: 11, batch: 150/188, acc_train: 0.767, acc_test: 0.789
    epoch_nr: 12, batch: 50/188, acc_train: 0.783, acc_test: 0.789
    epoch_nr: 12, batch: 100/188, acc_train: 0.768, acc_test: 0.762
    epoch_nr: 12, batch: 150/188, acc_train: 0.764, acc_test: 0.774
    epoch_nr: 13, batch: 50/188, acc_train: 0.776, acc_test: 0.786
    epoch_nr: 13, batch: 100/188, acc_train: 0.767, acc_test: 0.778
    epoch_nr: 13, batch: 150/188, acc_train: 0.764, acc_test: 0.785
    epoch_nr: 14, batch: 50/188, acc_train: 0.777, acc_test: 0.786
    epoch_nr: 14, batch: 100/188, acc_train: 0.764, acc_test: 0.765
    epoch_nr: 14, batch: 150/188, acc_train: 0.767, acc_test: 0.767
    epoch_nr: 15, batch: 50/188, acc_train: 0.778, acc_test: 0.786
    epoch_nr: 15, batch: 100/188, acc_train: 0.769, acc_test: 0.783
    epoch_nr: 15, batch: 150/188, acc_train: 0.770, acc_test: 0.783
    epoch_nr: 16, batch: 50/188, acc_train: 0.783, acc_test: 0.790
    epoch_nr: 16, batch: 100/188, acc_train: 0.766, acc_test: 0.787
    epoch_nr: 16, batch: 150/188, acc_train: 0.767, acc_test: 0.784
    
