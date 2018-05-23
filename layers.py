import keras
from keras import backend as K
import tensorflow as tf    

class ReducerSum(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReducerSum, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        #will pass a mask only if all entry were masked 
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.all(mask, axis=-1, keepdims=False)
        else: #don't return mask if not enough dimsions
            return None     
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)
    def call(self, inputs, mask = None):
        #only operate on last axis
        if mask is not None:
            mask = K.cast(mask, 'float32')
            mask = K.expand_dims(mask, axis=-1)
            return K.sum(inputs*mask, axis=-2)
        else:
            return K.sum(inputs, axis=-2)
        
class ReducerMean(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReducerMean, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.any(mask, axis=-1, keepdims=False)
        else: #don't return mask
            return None     
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)
    def call(self, inputs, mask = None):
        #only operate on last axis
        if mask is not None:
            mask = K.cast(mask, 'float32')
            #add axis for broadcasting
            mask = K.expand_dims(mask, axis=-1)
            return K.sum(inputs*mask, axis=-2) / (K.sum(mask, axis=-2) +K.epsilon())
        else:
            return K.mean(inputs, axis=-2)
        
class ReducerMax(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReducerMax, self).__init__(**kwargs)
        self.supports_masking = True
        
    def compute_mask(self, inputs, mask=None):
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.all(mask, axis=-1, keepdims=False)
        else: #don't return mask
            return None    
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)
    def call(self, inputs, mask = None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            mask = K.expand_dims(mask, axis=-1)
            return K.max(inputs*mask, axis=-2)
        else:
            return K.max(inputs, axis=-2)


class Dropper(keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropper, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = min(1., max(0., rate))
        self.seed = seed
        self.noise_shape = noise_shape
        
    def compute_mask(self, inputs, mask=None):
        return mask
    def _get_noise_shape(self, inputs):
        #only operate on last axis
        return self.noise_shape if self.noise_shape else K.concatenate([K.shape(inputs)[:-1], K.ones(1, dtype='int32')])
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs, mask = None, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            #zero things out
            #want to not zerot hings out of nothing would not be zero
            def dropped_inputs(inputs=inputs, rate=self.rate, seed=self.seed):
                kept_idx = K.greater_equal(K.random_uniform(noise_shape,
                                                            seed=seed), rate)
                kept_idx = K.cast(kept_idx, K.floatx())
                return inputs*kept_idx
            
        
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs
    
#this layer combines a laye
class Attention(keras.layers.Layer):
    def __init__(self,
                 kernel_initializer=None,
                 bias_initializer=None,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.kernel_initializer = keras.initializers.get('glorot_uniform')
        self.bias_initializer = keras.initializers.get('zeros')
        
        
    def compute_mask(self, inputs, mask=None):
        if len(inputs.shape)  > 2 and mask is not None:
            mask = K.all(mask, axis=-1, keepdims=False)
        else: #don't return mask
            return None  
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.sigmoid_kernel = self.add_weight(shape=(input_dim, 1),
                                      initializer=self.kernel_initializer,
                                      name='sigmoid_bias',
                                      regularizer=None,
                                      constraint=None)
        self.sigmoid_bias = self.add_weight(shape=(1, ),
                                      initializer=self.bias_initializer,
                                      name='sigmoid_kernel',
                                      regularizer=None,
                                      constraint=None)
        
        self.tanh_kernel = self.add_weight(shape=(input_dim, input_dim),
                                      initializer=self.kernel_initializer,
                                      name='tanh_kernel',
                                      regularizer=None,
                                      constraint=None)
        self.tanh_bias = self.add_weight(shape=(input_dim, ),
                                           initializer=self.bias_initializer,
                                           name='tanh_bias',
                                           regularizer=None,
                                           constraint=None)

        self.input_spec = keras.engine.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
    
    def compute_output_shape(self, input_shape):
        return input_shape[0:-2] + (input_shape[-1],)

    def call(self, inputs, mask = None):
        #first, tanh
        attention = K.dot(inputs, self.tanh_kernel)
        attention = K.bias_add(attention, self.tanh_bias)
        #now tanh
        attention = keras.activations.tanh(attention)
        
        #now sigmoid attention vector
        attention = K.dot(attention, self.sigmoid_kernel)
        attention = K.bias_add(attention, self.sigmoid_bias)
        attention = keras.activations.sigmoid(attention)

        #if mask there, then apply (zero out masked)
        if mask is not None:
            mask = K.cast(mask, 'float32')
            #add axis for broadcasting
            mask = K.expand_dims(mask, axis=-1)
            attention = attention * mask

        attention = attention / (K.sum(attention, axis=-1, keepdims=True) +K.epsilon())
        return K.sum(inputs*attention, axis=-2) / (K.sum(attention, axis=-2) +K.epsilon())
    
       
#custom Keras layer fo average embedding using weights
class WeigtedAverage(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeigtedAverage, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        #ouput will have shape of batch_size * embedding_size
        return (input_shape[0][0], input_shape[0][2])

    def compute_mask(self, inputs, mask=None):
        #don't pass mask on
        return None
    def call(self, inputs, mask=None):
        #Assuming that weights are second in tuple...
        #Assumse that weight shave dimesnion: batch_size * seq_length
        #add new axis

        #need to add epsilon to make certain that we never divide by 0à
        #shoiuld look at jkeras code and see how they handle this
        #do they use these sums? or batch_ddot?
        avg = K.sum(inputs[0] * inputs[1], axis=1) / (K.sum(inputs[1], axis=1) + K.epsilon())

        return avg

class SparseEmbedding(keras.layers.Layer):
    """This layer takes two 2d SparseTensor as inputs
    and returns a dense 2d Tensor. It does this by embedding
    and then combining the first SparseTensor, using the second as weights.
    """
    def __init__(self, vocab_size, embedding_size,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 input_length=None,
                 combiner = "sum",
                 **kwargs):
        super(SparseEmbedding, self).__init__(**kwargs)

        if combiner not in ["sum", "mean", "sqrtn"]:
            ValueError('"combiner" must be one of "sum", "mean", "sqrtn"')
        
        self.combiner = combiner
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.vocab_size, self.embedding_size),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint)
        self.built = True

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (self.embedding_size,)
        return output_shape
    
    def call(self, inputs):
        #inputs should be a tuple, with first element is ids and second is weights
        out = tf.nn.embedding_lookup_sparse(self.embeddings, inputs[0], inputs[1], combiner=self.combiner)
        return out

    def get_config(self):
        config = {'vocab_size': self.vocab_size,
                  'embedding_size': self.embedding_size,
                  'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
                  'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint)}
        base_config = super(SparseEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))