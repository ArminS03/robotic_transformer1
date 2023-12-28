import tensorflow_datasets as tfds
import tensorflow as tf

dataset_builder = tfds.builder_from_directory(builder_dir='./datasets/0.1.0')
print(dataset_builder.info)
dataset_builder_episodic_dataset = dataset_builder.as_dataset(split='train')
ds = list(dataset_builder_episodic_dataset)
print(type(ds))
print(len(ds))

counter = 1
for x in dataset_builder_episodic_dataset:
    # y = x['steps']
    print(x.keys())
    # for i in y:
    #     print(i)
    #     break
    break
        # counter += 1

  
# dataset_builder.download_and_prepare(download_dir="./notebooks/dataset/")

# Specify the directory path where your dataset is located
# data_dir = 'gs://gresearch/robotics/viola/0.1.0'

# # Load the dataset using tfds.load
# (ds_train, ds_test), ds_info = tfds.load(
#     'viola',  # Replace 'your_dataset_name' with the actual dataset name
#     split=['train', 'test'],
#     data_dir=data_dir,
#     with_info=True,
#     download_and_prepare_kwargs={
#         'download_dir': "./notebooks/dataset/",
#     },
# )

# Now you can use ds_train and ds_test in your TensorFlow code
