import tensorflow as tf

def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''
    
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)
    
    # TODO

    pass

