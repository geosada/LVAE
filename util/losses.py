import tensorflow as tf
import numpy as np
import sys

eps = 1e-8

class LossFunctions(object):

    def __init__(self, layers, dataset, encoder):

        self.ls = layers
        self.d  = dataset
        self.encoder = encoder
        self.reconst_pixel_log_stdv = tf.get_variable("reconst_pixel_log_stdv", initializer=tf.constant(0.0))

    def get_loss_pyx(self, logit, y):

        loss = self._ce(logit, y)
        accur = self._accuracy(logit, y)
        return loss, accur

    def get_loss_pxz(self, x_reconst, x_original, pxz):
        if pxz == 'Bernoulli':
            #loss = tf.reduce_mean( tf.reduce_sum(self._binary_crossentropy(x_original, x_reconst),1)) # reconstruction term
            loss = tf.reduce_mean( self._binary_crossentropy(x_original, x_reconst)) # reconstruction term
        elif pxz == 'LeastSquare':
            x_reconst  = tf.reshape( x_reconst, (-1, self.d.img_size))
            x_original = tf.reshape( x_original, (-1, self.d.img_size))
            #loss = tf.sqrt(tf.square(tf.reduce_mean(tf.subtract(x_original, x_reconst))) + eps)
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_original, x_reconst))) + eps)
        elif pxz == 'PixelSoftmax':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_reconst, labels=tf.cast(x_original, dtype=tf.int32))) / (self.d.img_size * 256)
        elif pxz == 'DiscretizedLogistic':
            loss = -tf.reduce_mean( self._discretized_logistic(x_reconst, x_original))
        else:
            sys.exit('invalid argument')
        return loss

    def _binary_crossentropy(self, t,o):
        t = tf.reshape( t, (-1, self.d.img_size))
        o = tf.reshape( o, (-1, self.d.img_size))
        return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

    def _discretized_logistic(self, x_reconst, x_original, binsize=1/256.0):
        # https://github.com/openai/iaf/blob/master/tf_utils/
        scale = tf.exp(self.reconst_pixel_log_stdv)
        x_original = (tf.floor(x_original / binsize) * binsize - x_reconst) / scale

        logp = tf.log(tf.sigmoid(x_original + binsize / scale) - tf.sigmoid(x_original) + eps)

        shape = x_reconst.get_shape().as_list()
        if len(shape) == 2:   # 1d
            indices = (1,2,3)
        elif len(shape) == 4: # cnn as NHWC
            indices = (1)
        else:
            raise ValueError('shape of x is unexpected')

        return tf.reduce_sum(logp, indices)

    def get_loss_kl(self, m, _lambda=1.0 ):

        L = m.L
        Z_SIZES = m.Z_SIZES

        """ KL divergence KL( q(z_l) || p(z_0)) at each lyaer, where p(z_0) is set as N(0,I) """
        Lzs1 = [0]*L

        """ KL( q(z_l) || p(z_l)) to monitor the activities of latent variable units at each layer
             as Fig.4 in http://papers.nips.cc/paper/6275-ladder-variational-autoencoders.pdf """
        Lzs2 = [0]*L

        for l in range(L):
            d_mu, d_logsigma = m.d_mus[l], m.d_logsigmas[l]
            p_mu, p_logsigma = m.p_mus[l], m.p_logsigmas[l]
    
            d_sigma = tf.exp(d_logsigma)
            p_sigma = tf.exp(p_logsigma)
            d_sigma2, p_sigma2 = tf.square(d_sigma), tf.square(p_sigma)
       
            kl1 = 0.5*tf.reduce_sum( (tf.square(d_mu) + d_sigma2) - 2*d_logsigma, 1) - Z_SIZES[l]*.5
            kl2 = 0.5*tf.reduce_sum( (tf.square(d_mu - p_mu) + d_sigma2)/p_sigma2 - 2*tf.log((d_sigma/p_sigma) + eps), 1) - Z_SIZES[l]*.5
    
            Lzs1[l] = tf.reduce_mean( tf.maximum(_lambda, kl1 ))
            Lzs2[l] = tf.reduce_mean( kl2 )
    
        """ use only KL-divergence at the top layer, KL( z_L || z_0) as loss cost for optimaization  """
        loss = Lzs1[-1]
        #loss += tf.add_n(Lzs2)
        return Lzs1, Lzs2, loss
