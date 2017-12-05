import tensorflow as tf
import numpy as np
from layers import Layers
from losses import LossFunctions


class LVAE(object):

    def __init__(self, d, lr, lambda_z_wu, do_classify, use_kl=True):

        """ model architecture """
        self.MLP_SIZES = [512, 256, 256, 128, 128]
        self.Z_SIZES   = [64, 32, 32, 32, 32]
        self.L = L = len(self.MLP_SIZES)

        self.do_classify = do_classify 

        """ flags for regularizers """
        self.use_kl   = use_kl

        """ data and external toolkits """
        self.d  = d  # dataset manager
        self.ls = Layers()
        self.lf = LossFunctions(self.ls, d, self.encoder)

        """ placeholders defined outside"""
        self.lr  = lr
        self.lambda_z_wu = lambda_z_wu

        """ cache for mu and sigma """
        self.e_mus, self.e_logsigmas = [0]*L, [0]*L  # q(z_i+1 | z_i), bottom-up inference as Eq.7-9
        self.p_mus, self.p_logsigmas = [0]*L, [0]*L  # p(z_i | z_i+1), top-down prior as Eq.1-3
        self.d_mus, self.d_logsigmas = [0]*L, [0]*L  # q(z_i | .), bidirectional inference as Eq.17-19


    def encoder(self, x, is_train=True, do_update_bn=True):

        h = x
        for l in range(self.L):
            scope = 'Encode_L' + str(l)
            h = self.ls.dense(scope, h, self.MLP_SIZES[l])
            h = self.ls.bn(scope, h, is_train, do_update_bn, name=scope)
            h = tf.nn.elu(h)

            """ prepare for bidirectional inference """
            _, self.e_mus[l], self.e_logsigmas[l] = self.ls.vae_sampler(
                scope, h, self.Z_SIZES[l], tf.nn.softplus
            ) # Eq.13-15
        #return h
        return self.e_mus[-1]

    def decoder(self, is_train=True, do_update_bn=True ):

        for l in range(self.L-1, -1, -1):
            scope = 'Decoder_L' + str(l)

            if l == self.L-1:
                """ At the highest latent layer, mu & sigma are identical to those outputed from encoer.
                    And making actual z is not necessary for the highest layer."""
                mu, logsigma = self.e_mus[l], self.e_logsigmas[l]
                self.d_mus[l], self.d_logsigmas[l] = mu, logsigma

                z = self.ls.sampler(self.d_mus[l], tf.exp(self.d_logsigmas[l]))

                """ prior of z_L is set as standard Gaussian, N(0,I). """
                self.p_mus[l], self.p_logsigmas[l] = tf.zeros((mu.get_shape())), tf.zeros((logsigma.get_shape()))

            else:
                """ prior is developed from z of the above layer """
                _, self.p_mus[l], self.p_logsigmas[l] = self.ls.vae_sampler(
                                                        scope, z, self.Z_SIZES[l], tf.nn.softplus
                                                    ) # Eq.13-15

                z, self.d_mus[l], self.d_logsigmas[l] = self.ls.precision_weighted_sampler(
                        scope,
                        (self.e_mus[l], tf.exp(self.e_logsigmas[l])),
                        (self.p_mus[l], tf.exp(self.p_logsigmas[l]))
                    )  # Eq.17-19

        """ go out to the input space """
        _d = self.d
        x = self.ls.dense('bottom',  z,  _d.img_size, tf.nn.elu) # reconstructed input

        if _d.is_3d: x = tf.reshape(x, (-1, _d.h,_d.w,_d.c))

        return x

        
    def build_graph_train(self, x_l, y_l, x):

        o = dict()  # output
        loss = 0

        logit = self.encoder(x)
        x_reconst = self.decoder()

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            logit_l = self.encoder(x_l, is_train=True, do_update_bn=False)  # for pyx and vat loss computation

        """ Classification Loss """
        if self.do_classify:
            o['Ly'], o['accur'] = self.lf.get_loss_pyx(logit_l, y_l)
            loss += o['Ly']

        """ for visualizationc """
        o['z'], o['y'] = logit, y_l

        """ p(x|z) Reconstruction Loss """
        o['Lr'] = self.lf.get_loss_pxz(x_reconst, x, 'DiscretizedLogistic')
        loss += o['Lr']
        o['x']  = x
        o['cs'] = x_reconst

        """ VAE KL-Divergence Loss """
        if self.use_kl:
            o['KL1'], o['KL2'], o['Lz'] = self.lf.get_loss_kl(self, _lambda=10.0)
            loss += self.lambda_z_wu * o['Lz']
        else:
            o['KL1'], o['KL2'], o['Lz'] = tf.constant(0), tf.constant(0), tf.constant(0)

        """ set losses """
        o['loss'] = loss
        self.o_train = o

        """ set optimizer """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        #self.op = optimizer.minimize(loss)
        grads = optimizer.compute_gradients(loss)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                #g = tf.Print(g, [g], "g %s = "%(v))
                grads[i] = (tf.clip_by_norm(g,5),v) # clip gradients
            else:
                print('g is None:', v)
                v = tf.Print(v, [v], "v = ", summarize=10000)
        #for v in tf.all_variables(): print("%s : %s" % (v.name,v.get_shape()))
        self.op = optimizer.apply_gradients(grads) # return train_op


    def build_graph_test(self, x_l, y_l):

        o = dict()  # output
        loss = 0

        logit_l = self.encoder(x_l, is_train=False, do_update_bn=False)  # for pyx and vat loss computation

        """ classification loss """
        if self.do_classify:
            o['Ly'], o['accur'] = self.lf.get_loss_pyx(logit_l, y_l)
            loss += o['Ly']

        """ for visualizationc """
        o['z'], o['y'] = logit_l, y_l

        """ set losses """
        o['loss'] = loss
        self.o_test = o
