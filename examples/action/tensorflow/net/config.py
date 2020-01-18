class RNNConfig(object):
    embedding_dim = 64
    num_classes = 101
    num_layers= 2           # num hidden layers
    hidden_dim = 256        # num hidden
    rnn = 'gru'             # lstm 或 gru
    dropout_keep_prob = 0.8 # dropout keep prob
    learning_rate = 1e-3    #
    batch_size = 128         #
    print_per_batch = 100    # display
    save_per_batch = 10      # each how batch save to tensorboard
    keep_prob = 0.8
    trainable = True
    weight_decay = 0.0005


class CNNConfig(object):
    basemodel = 'net.inception_resnet_v2'  # cnn encoder model
    batch_size = 4  # num of images in one batch
    val_batch_size = 4  # validate batch size
    decay_size = 5000 # num of batch in one epoch
    nrof_max_epoch_iters = 200000  # max iters of epoch
    validate_every_n_epochs = 5000  # validata every num epochs
    gpu_memory_fraction = 0.8  # Upper bound on the amount of GPU memory that will be used by the process.
    resized_width = 299
    resized_height = 299
    keep_probability = 0.8
    weight_decay = 5e-4
    random_crop =True
    random_rotate = True
    random_flip = True
    use_fixed_image_standardization =True
    #  choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],help='The optimization algorithm
                        # to use', default='ADAGRAD')
    optimizer = 'ADAM'
    learning_rate_decay_epochs = 100
    # Number of epochs between learning rate decay.
    # 'Initial learning rate. If set to a negative value a learning rate ,
    # schedule can be specified in the file "learning_rate_schedule.txt"
    learning_rate = 0.0
    learning_rate_schedule_file = './config/learning_rate_schedule_classifier_ucf.txt'
    learning_rate_decay_factor = 1.0  # Learning rate decay factor.
    moving_average_decay = 0.9999  # Exponential decay for tracking of training parameters.
    embedding_size = 1024
