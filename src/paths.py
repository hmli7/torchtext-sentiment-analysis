import os

environment_path = "../../hw3p2-data-V2/"
train_data_path = os.path.join(environment_path,"wsj0_train.npy")
train_labels_path = os.path.join(environment_path,"wsj0_train_merged_labels.npy")
valid_data_path = os.path.join(environment_path,"wsj0_dev.npy")
valid_labels_path = os.path.join(environment_path,"wsj0_dev_merged_labels.npy")
test_data_path = os.path.join(environment_path,"transformed_test_data.npy")
# test_labels_path = os.path.join(environment_path,"wsj0_train_merged_labels.npy")

# model_path = "../model/model_dict.pt"

output_path = os.path.join('..',"outputs")