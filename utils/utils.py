import tensorflow as tf
from networks import CNN
from parser import args
import os


"""
Data loading, we add an offset to the labels of fashion mnist so that
the federated model can differentiate between the two datasets.
"""
def initialize_dataloaders():
    if(args.dataset == 'mnist'):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        x_train, x_test = x_train / 255.0, x_test / 255.0

    elif(args.dataset == 'fmnist'):
        fmnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train_old), (x_test, y_test_old) = fmnist.load_data()
        y_train = y_train_old + 10
        y_test = y_test_old + 10
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif(args.dataset == 'full'):
        mnist = tf.keras.datasets.mnist
        (m_x_train, m_y_train), (m_x_test, m_y_test) = mnist.load_data()
        m_x_train = tf.cast(m_x_train, dtype=tf.float32)
        m_x_test = tf.cast(m_x_test, dtype=tf.float32)
        m_x_train, m_x_test = m_x_train / 255.0, m_x_test / 255.0

        fmnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train_old), (x_test, y_test_old) = fmnist.load_data()
        y_train = y_train_old + 10
        y_test = y_test_old + 10
        x_train = tf.cast(x_train, dtype=tf.float32)
        x_test = tf.cast(x_test, dtype=tf.float32)
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = tf.concat([x_train, m_x_train],0)
        x_test = tf.concat([x_test, m_x_test],0)
        y_train = tf.concat([y_train, m_y_train],0)
        y_test = tf.concat([y_test, m_y_test],0)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).repeat(args.epochs).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.eval_batch_size)

    return train_ds, test_ds, y_test # return labels for per class accuracy


def initialize_dirs(exp_path):
    os.makedirs(exp_path, exist_ok=True)


def initialize_criterion():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)


def initialize_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=args.l_rate, beta_1=tf.Variable(0.9),
    beta_2=tf.Variable(0.999))


def initialize_checkpoint(network, optimizer, exp_path):
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0.0), best_train=tf.Variable(0.0),
                                    best_test= tf.Variable(0.0), optimizer=optimizer,
                                    net=network)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, exp_path, max_to_keep=5)
    if(checkpoint_manager.latest_checkpoint):
        print("Resuming training from %s..." % checkpoint_manager.latest_checkpoint)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    return checkpoint, checkpoint_manager
