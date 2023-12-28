"""
robotic transformer(https://github.com/google-research/robotics_transformer)的多节点分布式训练代码,
采用tensorflow2的distribute.MultiWorkerMirroredStrategy(https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy)进行分布式训练，使用加载rlds(https://github.com/google-research/rlds)数据的方式进行数据的读取
使用方法：
    python distribute_worker_train.py --args = param, Among them, args can be found in the code get_args()
"""

import os
import sys

sys.path.append("/home/nh_intern1/")
import transformer_network
from tensor2robot.utils import tensorspec_utils
from tf_agents.specs import tensor_spec
import time
import rlds_dataset_loader
import tensorflow as tf
import jax
import argparse
import json
import matplotlib.pyplot as plt
import tensorflow_hub as hub

# dataset_links = ["gs://gresearch/robotics/language_table/0.0.1/",  "gs://gresearch/robotics/language_table_sim/0.0.1/", 
# "gs://gresearch/robotics/language_table_blocktoblock_sim/0.0.1/", 
# "gs://gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1/",
# "gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1/",
# "gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim/0.0.1/",
# "gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1/",
# "gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1/",
# "gs://gresearch/robotics/language_table_separate_oracle_sim/0.0.1/"]

dataset_links = ["gs://gresearch/robotics/language_table/0.0.1/"]

dataset_link =  "+".join(dataset_links)

def get_args():
    parser = argparse.ArgumentParser(description='Get distributed training parameters')
    parser.add_argument('--single_gpu_batch_size', '-s', help='batch size for single gpu', default=1, type=int)
    parser.add_argument('--training_epoch', '-te', help='training epoch', default=10000, type=int)  # training epoch
    parser.add_argument('--log_step', '-ls', help='log step', default=10, type=int)
    parser.add_argument('--dataset_dirs', '-d', help='dataset path', default=dataset_link)
    parser.add_argument('--learning_rate', '-lr', help='learning rate', default=0.00001, type=float)  # learning rate
    parser.add_argument('--vocab_size', '-vs', help='vocab size for discretization', default=256, type=int)  # discrete dictionary size
    parser.add_argument('--dataset_episode_num', '-den', help='Amount of training data', default=20000, type=int)
    parser.add_argument('--loaded_checkpoints_dir', '-lcd', help='Model loading directory', default="/home/nh_intern1/robotics_transformer/rt_1_x_tf_trained_for_002272480_step", type=str)
    parser.add_argument('--save_model', '-sm', help='save model', default=True)
    parser.add_argument('--model_save_epoch', '-mse', help='save model at every num epoch', default=10, type=int)
    parser.add_argument('--checkpoints_saved_dir', '-csd', help='Model save directory', default="", type=str)
    args = parser.parse_args()
    return args


time_sequence_length = 6  # Constants, from the paper


def create_train_dataset(args, global_batch_size):
    '''Create a dataset'''
    # dataset_dirs = args.dataset_dirs.split("+")

    workdir = "~/"
    sequence_length = time_sequence_length
    # data_target_width = 456
    # data_target_height = 256
    # random_crop_factor = 0.95
    replay_capacity = 5_000
    seed = 42
    rng = jax.random.PRNGKey(seed)
    rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())

    train_ds = rlds_dataset_loader.create_ds(
        data_rng,
        # dataset_dirs=dataset_dirs,
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        # target_width=data_target_width,
        # target_height=data_target_height,
        # random_crop_factor=random_crop_factor,
        cache=False,
        shuffle=True,
        shuffle_buffer_size=replay_capacity,
        cache_dir=workdir,
        # dataset_episode_num=args.dataset_episode_num
    )

    return train_ds


def create_model(args):
    '''Create model'''
    data_target_width = 320
    data_target_height = 256

    state_spec = tensorspec_utils.TensorSpecStruct()

    state_spec.image = tensor_spec.BoundedTensorSpec([data_target_height, data_target_width, 3],
                                                     dtype=tf.uint8,
                                                     name='image',
                                                     minimum=0.,
                                                     maximum=255.)
    state_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[512], dtype=tf.float32, name='natural_language_embedding')

    action_spec = tensorspec_utils.TensorSpecStruct()

    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')

    action_spec.world_vector = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1.75, maximum=1.75, name='world_vector')
    
    action_spec.rotation_delta = tensor_spec.BoundedTensorSpec(
        (3,), dtype=tf.float32, minimum=-1.2, maximum=1.2, name='rotation_delta')
    
    action_spec.gripper_closedness_action = tensor_spec.BoundedTensorSpec(
        (1,), dtype=tf.float32, minimum=-1.0, maximum=1.0, name='gripper_closedness_action')


    action_order = ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']
    network = transformer_network.TransformerNetwork(
        input_tensor_spec=state_spec,
        output_tensor_spec=action_spec,
        vocab_size=int(args.vocab_size),
        token_embedding_size=512,
        num_layers=8,
        layer_size=128,
        num_heads=8,
        feed_forward_size=512,
        dropout_rate=0.1,
        time_sequence_length=time_sequence_length,
        crop_size=236,
        use_token_learner=True,
        action_order=action_order)
    return network


def set_env():
    '''To set distributed training environment variables, see https://www.tensorflow.org/guide/distributed_training?hl=zh-cn#TF_CONFIG'''
    worker_idx = 0
    swarm = []
    swarm.append("ip" + ":" + "port")
    cluster = {'worker': swarm}
    type = "worker"
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': cluster,
        'task': {'type': type, 'index': worker_idx}
    })


def train_one_step(model, observation_batch, label_batch, network_state, optimizer):
    '''single step training'''
    with tf.GradientTape() as tape:
        model.set_actions(label_batch)
        model.call(observation_batch, network_state=network_state, training=True)
        loss = tf.reduce_mean(model.get_actor_loss())
        gradients = tape.gradient(loss, model.trainable_variables,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        logging_info = model.get_aux_info()
        return loss, logging_info

if __name__ == '__main__':
    # os.environ.pop('TF_CONFIG', None)

    # args = get_args()

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     for k in range(len(physical_devices)):
    #         tf.config.experimental.set_memory_growth(physical_devices[k], True)
    # else:
    #     print("Not enough GPUs")
    #     exit("quit unexpectedly")

    # set_env()

    # options = tf.distribute.experimental.CommunicationOptions(
    #     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    # )
    # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(
    #     communication_options=options)
    # global_batch_size = args.single_gpu_batch_size * mirrored_strategy.num_replicas_in_sync

    os.environ.pop('TF_CONFIG', None)

    args = get_args()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
    else:
        print("Not enough GPUs")
        exit("quit unexpectedly")

    # Set up environment for single-GPU training
    # mirrored_strategy = tf.distribute.MirroredStrategy(
    #     devices=["/gpu:0"]  # Use only the first GPU
    # )
    global_batch_size = args.single_gpu_batch_size # * mirrored_strategy.num_replicas_in_sync

    global_learning_rate = args.learning_rate * global_batch_size

    # with mirrored_strategy.scope():
    network = create_model(args)
    network.create_variables()
    dataset_dirs = args.dataset_dirs.split("+")
    train_ds = create_train_dataset(args, global_batch_size)
    network_state = tensor_spec.sample_spec_nest(
        network.state_spec, outer_dims=[args.single_gpu_batch_size])
    optimizer = tf.keras.optimizers.Adam(learning_rate=global_learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=network)
    
    if tf.train.latest_checkpoint(args.loaded_checkpoints_dir):
        status = ckpt.restore(tf.train.latest_checkpoint(args.loaded_checkpoints_dir))
        print(f"status: {status}" )
        print("restore model from %s" % (args.loaded_checkpoints_dir))
    
    current_step = ckpt.step.numpy()
    print("Start training")
    T1 = time.time()
    
    # action_order = network._action_order
    for epoch in range(1, args.training_epoch):
        total_loss = 0.0
        step = 0
        T1 = time.time()
        counter = 0
        # for i, data in enumerate(train_ds):
        #     counter += 1
        # print(counter)
        for i, data in enumerate(train_ds):
            if i % 500 == 0:
                print(f"step: {i}")
            # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            train_observation = data["observation"]
            # train_observation['image'] = tf.squeeze(train_observation['image'])
            # train_observation['natural_language_embedding'] = tf.squeeze(train_observation['natural_language_embedding'])
            train_labels = data["action"]
            '''single step training'''
            with tf.GradientTape() as tape:
                network.set_actions(train_labels)
                # print(train_observation)
                network.call(train_observation, network_state=network_state, training=True)
                loss = tf.reduce_mean(network.get_actor_loss())
                gradients = tape.gradient(loss, network.trainable_variables,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
                optimizer.apply_gradients(grads_and_vars=zip(gradients, network.trainable_variables))
                logging_info = network.get_aux_info()
        
            # per_replica_losses, logging_info = train_one_step(network, train_observation, train_labels, network_state, optimizer)
            per_replica_losses = tf.stop_gradient(loss)
            step = step + 1
            mean_loss = tf.reduce_mean(per_replica_losses)
            total_loss = total_loss + mean_loss
            ckpt.step.assign_add(1)
            # print('How long does it take to train one step?: %s s' % (10))
        T2 = time.time()
        print('Total time spent training for 1 epoch: ', ((T2 - T1)/60))
        if epoch % args.model_save_epoch == 0 and args.save_model:
            checkpoint_prefix = os.path.join(args.checkpoints_saved_dir, "ckpt")
            ckpt.save(checkpoint_prefix)
            print("Model save location: %s !" % (checkpoint_prefix))

print("Exit normally!")


def instruction_embedding(instr):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    # print ("module %s loaded" % module_url)
    return model(instr)