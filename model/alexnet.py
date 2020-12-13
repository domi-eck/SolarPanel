import tensorflow as tf

fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)

def create(x, num_outputs, dropout_rate = 0.5):
    '''
        args:
            x               network input
            num_outputs     number of logits
            dropout         dropout rate during training

            Conv2D(96, 11, 4)
            BatchNorm()
            ReLU()

            M axP ool(3, 2)
            Conv2D(192, 5, 1)
            BatchNorm()
            ReLU ()

            M axP ool(3, 2)
            Conv2D(384, 3, 1)
            BatchNorm()
            ReLU ()


            Conv2D(256, 3, 1)
            BatchNorm()
            ReLU()


            Conv2D(256, 3, 1)
            BatchNorm()
            ReLU()


            M axP ool(3, 2)
            DropOut()
            FC(4096)
            ReLU()


            DropOut()
            F C(4096)
            ReLU()
            F C(2)
    '''
    
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)


    # 1st convolutional layer
    conv1 = tf.layers.conv2d(x, filters=96, kernel_size = 11, strides =4, padding="same", use_bias=True)
    conv1 = tf.layers.batch_normalization(conv1, training = is_training)
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.layers.max_pooling2d(conv1, 3, 2)
    conv2 = tf.layers.conv2d(conv2, filters=192, kernel_size=5, strides=1, padding="same", use_bias=True )
    conv2 = tf.layers.batch_normalization(conv2, training = is_training)
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.layers.max_pooling2d(conv2, 3, 2)
    conv3 = tf.layers.conv2d(conv3, filters=384, kernel_size=3, strides=1, padding="same", use_bias=True)
    conv3 = tf.layers.batch_normalization(conv3, training = is_training)
    conv3 = tf.nn.relu(conv3)

    conv4 = tf.layers.conv2d(conv3, filters=256, kernel_size=3, strides=1, padding="same", use_bias=True)
    conv4 = tf.layers.batch_normalization(conv4, training = is_training)
    conv4 = tf.nn.relu(conv4)

    conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=3, strides=1, padding="same", use_bias=True)
    conv5 = tf.layers.batch_normalization(conv5, training = is_training)

    conv6 = tf.layers.max_pooling2d(conv5, 3, 2)
    conv6 = tf.layers.flatten(conv6)
    conv6 = tf.layers.dropout(conv6, rate=0.5, training=is_training)
    conv6 = tf.contrib.layers.fully_connected(conv6, 4096)


    conv7 = tf.layers.dropout(conv6, rate=0.2, training = is_training)
    conv7 = tf.contrib.layers.fully_connected(conv7, 4096)
    conv7 = tf.contrib.layers.fully_connected(conv7, 2, activation_fn=None)


    return(conv7)


