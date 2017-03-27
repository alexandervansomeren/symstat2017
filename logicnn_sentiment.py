import cPickle
import pprint

import numpy as np
from collections import defaultdict, OrderedDict
import theano
# theano.config.cxx = '/usr/local/bin/gcc-6'
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import re
import warnings
import sys
import time
import os
import math

warnings.filterwarnings("ignore")


def Iden(x):
    y = x
    return (y)


def train_conv_net(datasets,
                   U,
                   word_idx_map,
                   img_w=300,
                   filter_hs=[3, 4, 5],
                   hidden_units=[100, 2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=11,
                   batch_size=50,
                   lr_decay=0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   pi_params=[1., 0],
                   C=1.0,
                   patience=20):
    """
    Train a convnet through iterative distillation
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper [Kim, 2014]
    lr_decay = adadelta decay parameter
    pi_params = update strategy of imitation parameter \pi
    C = regularization strength
    patience = number of iterations without performance improvement before stopping
    """
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0]) - 1
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))
    parameters = [("image shape", img_h, img_w), ("filter shape", filter_shapes), ("hidden_units", hidden_units),
                  ("dropout", dropout_rate), ("batch_size", batch_size), ("non_static", non_static),
                  ("learn_decay", lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static),
                  ("sqr_norm_lim", sqr_norm_lim), ("shuffle_batch", shuffle_batch), ("pi_params", pi_params), ("C", C)]
    print parameters

    # define model architecture
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value=U, name="Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))],
                               allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, img_h, img_w),
                                        filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs, 1)
    hidden_units[0] = feature_maps * len(filter_hs)
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations,
                            dropout_rates=dropout_rate)

    # build the feature of BUT-rule
    f_but = T.fmatrix('f_but')
    f_but_ind = T.fmatrix('f_ind')  # indicators
    f_but_layer0_input = Words[T.cast(f_but.flatten(), dtype="int32")].reshape(
        (f_but.shape[0], 1, f_but.shape[1], Words.shape[1]))
    f_but_pred_layers = []
    for conv_layer in conv_layers:
        f_but_layer0_output = conv_layer.predict(f_but_layer0_input, batch_size)
        f_but_pred_layers.append(f_but_layer0_output.flatten(2))
    f_but_layer1_input = T.concatenate(f_but_pred_layers, 1)
    f_but_y_pred_p = classifier.predict_p(f_but_layer1_input)
    f_but_full = T.concatenate([f_but_ind, f_but_y_pred_p], axis=1)  # batch_size x 1 + batch_size x K
    f_but_full = theano.gradient.disconnected_grad(f_but_full)

    # add logic layer
    nclasses = 2
    rules = [FOL_But(nclasses, x, f_but_full)]
    rule_lambda = [1]
    new_pi = get_pi(cur_iter=0, params=pi_params)
    logic_nn = LogicNN(rng, input=x, network=classifier, rules=rules, rule_lambda=rule_lambda, pi=new_pi, C=C)

    # define parameters of the model and update functions using adadelta
    params_p = logic_nn.params_p
    for conv_layer in conv_layers:
        params_p += conv_layer.params
    if non_static:
        # if word vectors are allowed to change, add them as model parameters
        params_p += [Words]
    cost_p = logic_nn.negative_log_likelihood(y)
    dropout_cost_p = logic_nn.dropout_negative_log_likelihood(y)
    grad_updates_p = sgd_updates_adadelta(params_p, dropout_cost_p, lr_decay, 1e-6, sqr_norm_lim)

    # shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    # extra data (at random)
    np.random.seed(3435)
    # training data
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        # shuffle both train data and features
        permutation_order = np.random.permutation(datasets[0].shape[0])
        train_set = datasets[0][permutation_order]
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[0], extra_data, axis=0)
        new_fea = {}
        train_fea = datasets[3]
        for k in train_fea.keys():
            train_fea_k = train_fea[k][permutation_order]
            extra_fea = train_fea_k[:extra_data_num]
            new_fea[k] = np.append(train_fea[k], extra_fea, axis=0)
        train_text = datasets[6][permutation_order]
        extra_text = train_text[:extra_data_num]
        new_text = np.append(datasets[6], extra_text, axis=0)
    else:
        new_data = datasets[0]
        new_fea = datasets[3]
        new_text = datasets[6]
    # shuffle both training data and features
    permutation_order = np.random.permutation(new_data.shape[0])
    new_data = new_data[permutation_order]
    for k in new_fea.keys():
        new_fea[k] = new_fea[k][permutation_order]
    new_text = new_text[permutation_order]
    n_batches = new_data.shape[0] / batch_size
    n_train_batches = n_batches
    train_set = new_data
    train_set_x, train_set_y = shared_dataset((train_set[:, :img_h], train_set[:, -1]))
    train_fea = new_fea
    train_fea_but_ind = train_fea['but_ind'].reshape([train_fea['but_ind'].shape[0], 1])
    train_fea_but_ind = shared_fea(train_fea_but_ind)
    for k in new_fea.keys():
        if k != 'but_text':
            train_fea[k] = shared_fea(new_fea[k])

    # val data
    if datasets[1].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[1].shape[0] % batch_size
        # shuffle both val data and features
        permutation_order = np.random.permutation(datasets[1].shape[0])
        val_set = datasets[1][permutation_order]
        extra_data = val_set[:extra_data_num]
        new_val_data = np.append(datasets[1], extra_data, axis=0)
        new_val_fea = {}
        val_fea = datasets[4]
        for k in val_fea.keys():
            val_fea_k = val_fea[k][permutation_order]
            extra_fea = val_fea_k[:extra_data_num]
            new_val_fea[k] = np.append(val_fea[k], extra_fea, axis=0)
        val_text = datasets[7][permutation_order]
        extra_text = val_text[:extra_data_num]
        new_val_text = np.append(datasets[7], extra_text, axis=0)
    else:
        new_val_data = datasets[1]
        new_val_fea = datasets[4]
        new_val_text = datasets[7]
    val_set = new_val_data
    val_set_x, val_set_y = shared_dataset((val_set[:, :img_h], val_set[:, -1]))
    n_batches = new_val_data.shape[0] / batch_size
    n_val_batches = n_batches
    val_fea = new_val_fea
    val_fea_but_ind = val_fea['but_ind'].reshape([val_fea['but_ind'].shape[0], 1])
    val_fea_but_ind = shared_fea(val_fea_but_ind)
    for k in val_fea.keys():
        if k != 'but_text':
            val_fea[k] = shared_fea(val_fea[k])

    # test data
    test_set_x = datasets[2][:, :img_h]
    test_set_y = np.asarray(datasets[2][:, -1], "int32")
    test_fea = datasets[5]
    test_fea_but_ind = test_fea['but_ind']
    test_fea_but_ind = test_fea_but_ind.reshape([test_fea_but_ind.shape[0], 1])
    test_text = datasets[8]

    ### compile theano functions to get train/val/test errors
    val_model = theano.function([index], logic_nn.errors(y),
                                givens={
                                    x: val_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: val_set_y[index * batch_size: (index + 1) * batch_size],
                                    f_but: val_fea['but'][index * batch_size: (index + 1) * batch_size],
                                    f_but_ind: val_fea_but_ind[index * batch_size: (index + 1) * batch_size, :]},
                                allow_input_downcast=True,
                                on_unused_input='warn')

    test_model = theano.function([index], logic_nn.errors(y),
                                 givens={
                                     x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: train_set_y[index * batch_size: (index + 1) * batch_size],
                                     f_but: train_fea['but'][index * batch_size: (index + 1) * batch_size],
                                     f_but_ind: train_fea_but_ind[index * batch_size: (index + 1) * batch_size, :]},
                                 allow_input_downcast=True,
                                 on_unused_input='warn')

    train_model = theano.function([index], cost_p, updates=grad_updates_p,
                                  givens={
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size],
                                      f_but: train_fea['but'][index * batch_size: (index + 1) * batch_size],
                                      f_but_ind: train_fea_but_ind[index * batch_size: (index + 1) * batch_size, :]},
                                  allow_input_downcast=True,
                                  on_unused_input='warn')

    ### setup testing
    test_size = test_set_x.shape[0]
    print 'test size ', test_size
    test_pred_layers = []
    test_layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((test_size, 1, img_h, Words.shape[1]))
    f_but_test_pred_layers = []
    f_but_test_layer0_input = Words[T.cast(f_but.flatten(), dtype="int32")].reshape(
        (test_size, 1, img_h, Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
        f_but_test_layer0_output = conv_layer.predict(f_but_test_layer0_input, test_size)
        f_but_test_pred_layers.append(f_but_test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    f_but_test_layer1_input = T.concatenate(f_but_test_pred_layers, 1)
    f_but_test_y_pred_p = classifier.predict_p(f_but_test_layer1_input)
    f_but_test_full = T.concatenate([f_but_ind, f_but_test_y_pred_p], axis=1)  # Ns x 1 + Ns x K

    # transform to shared variables
    test_set_x_shr, test_set_y_shr = shared_dataset((test_set_x, test_set_y))

    test_q_y_pred, test_p_y_pred = logic_nn.predict(test_layer1_input,
                                                    test_set_x_shr,
                                                    [f_but_test_full])
    test_q_error = T.mean(T.neq(test_q_y_pred, y))
    test_p_error = T.mean(T.neq(test_p_y_pred, y))
    test_model_all = theano.function([x, y, f_but, f_but_ind],
                                     [test_q_error, test_p_error], allow_input_downcast=True,
                                     on_unused_input='warn')

    ### start training over mini-batches
    print '... training'
    epoch = 0
    batch = 0
    best_val_q_perf = 0
    val_p_perf = 0
    val_q_perf = 0
    cost_epoch = 0
    stop_count = 0
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        # train
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                batch = batch + 1
                new_pi = get_pi(cur_iter=batch * 1. / n_train_batches, params=pi_params)
                logic_nn.set_pi(new_pi)
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                batch = batch + 1
                new_pi = get_pi(cur_iter=batch * 1. / n_train_batches, params=pi_params)
                logic_nn.set_pi(new_pi)
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        # eval
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_losses = np.array(train_losses)
        train_q_perf = 1 - np.mean(train_losses[:, 0])
        train_p_perf = 1 - np.mean(train_losses[:, 1])
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_losses = np.array(val_losses)
        val_q_perf = 1 - np.mean(val_losses[:, 0])
        val_p_perf = 1 - np.mean(val_losses[:, 1])
        print(
        'epoch: %i, training time: %.2f secs; (q): train perf: %.4f %%, val perf: %.4f %%; (p): train perf: %.4f %%, val perf: %.4f %%' % \
        (epoch, time.time() - start_time, train_q_perf * 100., val_q_perf * 100., train_p_perf * 100.,
         val_p_perf * 100.))
        test_loss = test_model_all(test_set_x, test_set_y, test_fea['but'], test_fea_but_ind)
        test_loss = np.array(test_loss)
        test_perf = 1 - test_loss
        print 'test perf: q %.4f %%, p %.4f %%' % (test_perf[0] * 100., test_perf[1] * 100.)
        if val_q_perf > best_val_q_perf:
            best_val_q_perf = val_q_perf
            ret_test_perf = test_perf
            stop_count = 0
        else:
            stop_count += 1
        if stop_count == patience:
            break
    return ret_test_perf


def get_pi(cur_iter, params=None, pi=None):
    """ exponential decay: pi_t = max{1 - k^t, lb} """
    k, lb = params[0], params[1]
    pi = 1. - max([k ** cur_iter, lb])
    return pi


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def shared_fea(fea, borrow=True):
    """ 
    Function that loads the features into shared variables
    """
    shared_fea = theano.shared(np.asarray(fea,
                                          dtype=theano.config.floatX),
                               borrow=borrow)
    return shared_fea


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        if param.name == 'Words':
            stepped_param = param + step * .3
        else:
            stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


# 'but'-rule feature
def get_idx_from_but_fea(but_fea, but_ind, word_idx_map, max_l=51, k=300, filter_h=5):
    if but_ind == 0:
        pad = filter_h - 1
        x = [0] * (max_l + 2 * pad)
    else:
        x = get_idx_from_sent(but_fea, word_idx_map, max_l, k, filter_h)
    return x


def make_idx_data(revs, fea, word_idx_map, max_l=51, k="Not used!", filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, dev, test = [], [], []
    train_text, dev_text, test_text = [], [], []
    train_fea, dev_fea, test_fea = {}, {}, {}
    fea['but'] = []
    for k in fea.keys():
        train_fea[k], dev_fea[k], test_fea[k] = [], [], []
    for i, rev in enumerate(revs):
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, "Not used!", filter_h)
        sent.append(rev["y"])
        fea['but'].append(get_idx_from_but_fea(fea['but_text'][i], fea['but_ind'][i], word_idx_map, max_l, k, filter_h))
        if rev["split"] == 0:
            train.append(sent)
            for k, v in fea.iteritems():
                train_fea[k].append(v[i])
            train_text.append(rev["text"])
        elif rev["split"] == 1:
            dev.append(sent)
            for k, v in fea.iteritems():
                dev_fea[k].append(v[i])
            dev_text.append(rev["text"])
        else:
            test.append(sent)
            for k, v in fea.iteritems():
                test_fea[k].append(v[i])
            test_text.append(rev["text"])
    train = np.array(train, dtype="int")
    dev = np.array(dev, dtype="int")
    test = np.array(test, dtype="int")
    for k in fea.keys():
        if k == 'but':
            train_fea[k] = np.array(train_fea[k], dtype='int')
            dev_fea[k] = np.array(dev_fea[k], dtype='int')
            test_fea[k] = np.array(test_fea[k], dtype='int')
        elif k == 'but_text':
            train_fea[k] = np.array(train_fea[k])
            dev_fea[k] = np.array(dev_fea[k])
            test_fea[k] = np.array(test_fea[k])
        else:
            train_fea[k] = np.array(train_fea[k], dtype=theano.config.floatX)
            dev_fea[k] = np.array(dev_fea[k], dtype=theano.config.floatX)
            test_fea[k] = np.array(test_fea[k], dtype=theano.config.floatX)
    train_text = np.array(train_text)
    dev_text = np.array(dev_text)
    test_text = np.array(test_text)
    return [train, dev, test, train_fea, dev_fea, test_fea, train_text, dev_text, test_text]


if __name__ == "__main__":
    path = 'data/'
    print path
    print "loading data...",
    x = cPickle.load(open("%s/stsa.binary.p" % path, "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

    # for rev in revs[0:50]:
    #     pprint.pprint(rev)

    ################################
    # rev looks like
    # {
    #   'num_words': 16,                    ##  number of words
    #   'split': 0,                         ##  0: train, 1: dev, 2: test
    #   'text': 'make a splash even         ##  text of snippet (somehow subparts of this sentence are also training
    #           greater than arnold             examples!)
    #           schwarzenegger ,
    #           jean claud van damme
    #           or steven segal',
    #   'y': 1                              ##  sentiment (positive or negative)
    # }

    # {'num_words': 1, 'split': 0, 'text': 'splash', 'y': 1}
    # {'num_words': 2, 'split': 0, 'text': 'a splash', 'y': 1}
    # {'num_words': 1, 'split': 0, 'text': 'greater', 'y': 1}
    # {'num_words': 4, 'split': 0, 'text': 'a splash even greater', 'y': 1}
    # {'num_words': 5, 'split': 0, 'text': 'make a splash even greater', 'y': 1}
    # {'num_words': 2, 'split': 0, 'text': 'arnold schwarzenegger', 'y': 0}

    ################################
    # W :           w2v vectors -- len: 17237
    ################################
    # W2:           are random vectors -- len:17237
    ################################
    # word_idx_map

    print word_idx_map['unimaginative']  # {'unimaginative': 1}
    print vocab.values()[0:100]                     # [0.0, 0.0,
    print "\n\n\n\n\n"


    print "data loaded!"
    print "loading features..."
    fea = cPickle.load(open("%s/stsa.binary.p.fea.p" % path, "rb"))

    print "features loaded!"

    mode = sys.argv[1]
    word_vectors = sys.argv[2]
    if mode == "-nonstatic":
        print "model architecture: CNN-non-static"
        non_static = True
    elif mode == "-static":
        print "model architecture: CNN-static"
        non_static = False
    if word_vectors == "-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors == "-word2vec":
        print "using: word2vec vectors, dim=%d" % W.shape[1]
        U = W

    execfile("logicnn_classes.py")
    # from logicnn_classes import *

    # q: teacher network; p: student network
    q_results = []
    p_results = []
    datasets = make_idx_data(revs, fea, word_idx_map, max_l=53, k="Not used!", filter_h=5)
    # train_size = 25000
    # datasets[0] = datasets[0][:train_size]  # 76961
    # datasets[6] = datasets[6][:train_size]
    indices = [3633, 5528, 3001, 5453, 4580, 3467, 1375, 3759, 5553, 3414, 1366, 1461, 3776, 3044, 1676, 6092, 3822, 40,
               6075, 481, 576, 3586, 3183, 64, 521, 3777, 5723, 1004, 5026, 1518, 4289, 6784, 2593, 5431, 1725, 6465,
               4154, 614, 3114, 3940, 3821, 2809, 1489, 4006, 1467, 5107, 4055, 5213, 3332, 5820, 3070, 2185, 3728, 43,
               6830, 5690, 1309, 5676, 5954, 3998, 1892, 5666, 3348, 5387, 880, 6705, 3466, 4316, 3526, 6266, 6061,
               2315, 3532, 6107, 2093, 4985, 4978, 2688, 164, 6223, 3695, 1863, 1488, 285, 5259, 199, 4244, 2190, 1359,
               5330, 900, 831, 2963, 5684, 5699, 6149, 895, 4972, 6210, 5031, 5507, 2217, 1071, 4087, 1822, 2425, 5512,
               3985, 4738, 249, 1037, 5217, 5974, 3282, 621, 2815, 507, 1177, 5625, 1173, 1215, 3591, 6131, 5757, 3244,
               721, 6243, 3019, 4337, 4609, 5942, 3608, 2964, 3592, 6873, 2848, 6175, 3870, 3360, 6798, 6534, 3948,
               3258, 2373, 4779, 355, 573, 2857, 1378, 361, 4323, 3010, 857, 4434, 2253, 1794, 1255, 5443, 808, 1647,
               2450, 478, 6096, 4065, 5868, 2380, 5133, 533, 5722, 5712, 2409, 244, 4189, 1498, 4886, 1203, 1092, 5073,
               3471, 295, 3447, 1306, 5365, 2969, 4001, 6233, 6183, 1893, 3395, 1724, 4109, 3007, 1063, 6426, 702, 6961,
               4427, 1885, 2346, 2312, 6788, 4818, 2831, 2589, 1780, 5947, 826, 1758, 5480, 5707, 618, 5253, 4622, 6990,
               4697, 1010, 1995, 5849, 1321, 4647, 6706, 4628, 6176, 1908, 520, 240, 2733, 4520, 5455, 4019, 6163, 836,
               1720, 784, 6877, 336, 104, 1185, 179, 6624, 4888, 71, 1129, 473, 52, 2118, 4573, 6731, 6580, 6341, 5076,
               5239, 6238, 4824, 1027, 172, 2946, 3654, 3327, 5233, 3860, 1609, 5934, 352, 4758, 1348, 4325, 4527, 2251,
               2347, 2480, 5205, 3260, 121, 5361, 4110, 647, 4255, 5754, 2788, 3839, 1967, 3272, 1645, 426, 6429, 6808,
               6015, 5751, 1199, 143, 3648, 5160, 5125, 3071, 3157, 819, 3769, 5234, 6424, 5688, 6974, 3301, 3306, 6445,
               1464, 3984, 1332, 6326, 4927, 1965, 2894, 6402, 5991, 2412, 2354, 2711, 3118, 6740, 4460, 5656, 5349,
               3186, 4373, 3893, 118, 3506, 6752, 2261, 1479, 309, 5713, 41, 6169, 5802, 5904, 134, 6145, 3972, 1728,
               4999, 1563, 1812, 5164, 5129, 864, 953, 1560, 5137, 4271, 1864, 465, 1655, 5405, 4510, 3724, 4941, 2405,
               3990, 6203, 6661, 4653, 1471, 6570, 2541, 2666, 5738, 1162, 4076, 1741, 2200, 4439, 24, 4969, 373, 5844,
               1179, 5930, 2925, 4911, 2055, 4387, 6742, 47, 5567, 792, 5040, 2272, 3113, 5929, 3231, 6658, 1076, 4399,
               1627, 2397, 5628, 5175, 4022, 5693, 2085, 2234, 3992, 1891, 2770, 5542, 2692, 1339, 4678, 1395, 1989,
               5368, 658, 2514, 6327, 435, 6509, 5548, 6994, 1327, 2062, 1659, 2209, 5560, 2194, 3281, 4535, 2970, 4374,
               5861, 1628, 5083, 6263, 4088, 3178, 6832, 359, 4904, 5086, 103, 4879, 1973, 2673, 2259, 5918, 585, 1613,
               5245, 4149, 3863, 1224, 3342, 1987, 6530, 3755, 261, 2005, 4615, 730, 1079, 4421, 5399, 4683, 2865, 892,
               3440, 988, 5695, 358, 3376, 5921, 1286, 5078, 1988, 6545, 1719, 5146, 4993, 2722, 4114, 2936, 4190, 3223,
               722, 4456, 4709, 845, 740, 1606, 1189, 4854, 1625, 4516, 4892, 720, 5371, 4769, 6058, 1747, 462, 6910,
               4073, 402, 6782, 5181, 1925, 2898, 1083, 4825, 1044, 1394, 4621, 2566, 5134, 388, 4676, 2014, 5166, 3333,
               1621, 6687, 4355, 610, 6082, 5390, 3706, 6146, 3122, 68, 6554, 3528, 5853, 183, 3937, 4050, 6375, 6313,
               4431, 342, 5065, 5433, 3963, 3075, 1323, 276, 5124, 5531, 3098, 228, 5500, 6634, 188, 736, 3009, 1123,
               4744, 5054, 4062, 6552, 3496, 6342, 3427, 3065, 6406, 4493, 4968, 5373, 5139, 503, 4701, 4745, 2765,
               4819, 946, 4816, 6295, 2622, 4572, 113, 6063, 6024, 3630, 283, 1133, 5508, 2901, 4468, 939, 2620, 1778,
               5663, 821, 1590, 1711, 5620, 5714, 3357, 4736, 5289, 5252, 2220, 1505, 1422, 6722, 3692, 222, 4618, 4101,
               6393, 6805, 4466, 904, 3716, 2313, 1982, 6556, 4748, 5999, 2880, 4208, 1859, 3568, 4860, 2342, 3041,
               3170, 1555, 3476, 4530, 367, 4287, 4070, 1565, 4908, 5680, 196, 776, 1880, 6303, 3647, 9, 2319, 3736,
               3946, 1230, 3495, 4058, 4971, 1110, 5174, 4080, 83, 3677, 1993, 1641, 204, 3573, 6898, 4170, 5329, 3449,
               5788, 6504, 2198, 6354, 4801, 930, 2485, 3731, 5544, 1048, 6247, 3034, 6772, 1223, 2678, 3874, 2664,
               6845, 2725, 6767, 6908, 2981, 450, 4115, 1466, 4560, 1022, 2229, 5478, 4600, 5749, 5121, 4532, 3763,
               4204, 5541, 970, 1651, 1594, 2408, 3421, 815, 6942, 4369, 1014, 532, 6330, 2396, 5071, 5607, 2895, 4753,
               2956, 6897, 3739, 1211, 3744, 5300, 4994, 2068, 2861, 4097, 2333, 3358, 1169, 2723, 59, 6906, 1900, 4632,
               3208, 4327, 63, 1946, 3882, 4916, 3064, 1792, 3709, 5020, 1106, 2997, 2792, 5519, 6753, 6817, 6711, 4790,
               733, 2035, 3578, 4789, 2286, 6314, 6560, 5936, 2355, 2076, 1897, 1538, 1392, 3823, 6890, 5594, 6727,
               4811, 3799, 5790, 4949, 6694, 1065, 5672, 3004, 4040, 5262, 2778, 2285, 6023, 921, 6521, 4899, 5364,
               5730, 552, 6905, 2050, 5559, 5075, 1562, 4610, 1446, 3749, 6559, 4129, 2440, 3365, 727, 1008, 125, 33,
               6140, 3247, 4617, 4976, 3524, 2598, 619, 5135, 6756, 5772, 3894, 3322, 1752, 2702, 3109, 3155, 3668,
               3547, 4237, 5278, 4685, 1096, 2756, 949, 4543, 6030, 3535, 322, 3595, 5783, 2310, 3700, 5669, 5956, 5925,
               1212, 4199, 6695, 5755, 1941, 960, 3450, 6849, 6786, 6851, 4206, 2330, 3378, 896, 4269, 1517, 6312, 4638,
               4166, 6697, 4765, 4120, 661, 3904, 4624, 6535, 1035, 3691, 1580, 3806, 6794, 3825, 1057, 3553, 5123,
               3585, 1510, 4631, 793, 4850, 3664, 186, 6500, 385, 2535, 3106, 5815, 660, 2258, 5958, 1771, 1072, 3611,
               2709, 6894, 1040, 4402, 444, 5660, 4275, 4151, 4781, 6121, 4526, 5001, 3017, 927, 1883, 5979, 5041, 6443,
               6517, 6616, 3772, 6196, 6539, 1706, 6700, 6098, 4099, 2463, 4859, 851, 1541, 5984, 2482, 281, 5965, 3624,
               2019, 2353, 2955, 267, 263, 6225, 1634, 586, 2613, 841, 5706, 6359, 5305, 5887, 4328, 5410, 1640, 6887,
               1672, 1644, 38, 3655, 4537, 3412, 6298, 390, 2667, 82, 2224, 3284, 4333, 3356, 6188, 6362, 6293, 1006,
               674, 1887, 6345, 62, 2348, 3872, 5311, 5231, 1218, 5334, 4729, 1564, 1054, 5881, 4585, 6399, 6139, 2551,
               2152, 5401, 6274, 17, 2679, 5709, 1220, 4453, 536, 6608, 6913, 279, 4863, 5564, 4222, 1369, 790, 4194,
               6648, 830, 6759, 2160, 3074, 5381, 5527, 1762, 6033, 1968, 5632, 4430, 4500, 4029, 5489, 6672, 714, 332,
               5584, 2103, 6610, 5773, 3119, 2984, 4313, 5776, 4079, 2948, 2156, 1194, 3138, 511, 3062, 5854, 5848,
               1929, 3384, 4959, 5270, 6609, 1919, 5992, 6447, 4648, 3014, 2924, 6261, 5728, 4071, 6173, 1, 1814, 6373,
               5398, 4173, 3751, 2011, 6347, 6285, 1415, 3897, 3092, 5536, 639, 1159, 4267, 893, 6934, 3993, 2803, 1237,
               4740, 1445, 3536, 974, 3341, 4061, 6867, 6726, 4234, 2871, 5610, 3198, 4315, 2470, 1090, 218, 4462, 6837,
               375, 5736, 6548, 1751, 1248, 5639, 725, 107, 1282, 72, 5423, 1944, 6815, 5908, 6237, 5964, 206, 3815,
               3913, 1000, 6657, 6423, 2188, 5204, 3671, 5163, 965, 1581, 4229, 2572, 5, 6008, 3927, 2424, 2357, 3161,
               415, 513, 2007, 2652, 4563, 4635, 4300, 6754, 807, 2947, 302, 2633, 1226, 2368, 4556, 5801, 476, 4542,
               6901, 5715, 58, 347, 4341, 1236, 3451, 5140, 4593, 2431, 48, 3793, 4599, 872, 6585, 4084, 6968, 4741,
               4656, 6750, 3194, 6195, 5229, 6593, 6865, 5547, 2039, 1156, 161, 5188, 738, 5664, 2671, 6036, 2036, 6021,
               4996, 3314, 42, 3861, 951, 3761, 3514, 1205, 6814, 237, 4958, 4565, 202, 4977, 4590, 5483, 6027, 441,
               6114, 3326, 5739, 2052, 1373, 330, 3375, 2672, 1474, 607, 4448, 6635, 3286, 5977, 897, 1700, 1626, 5962,
               861, 3399, 5996, 5819, 5283, 6516, 5842, 6269, 1450, 2856, 3507, 3345, 4883, 2338, 2360, 3909, 2388,
               1746, 5255, 4921, 4566, 3296, 1234, 3615, 303, 3663, 4068, 5003, 6822, 1721, 2429, 4416, 5998, 2451, 978,
               3979, 122, 5877, 3540, 6895, 3730, 2755, 4990, 4386, 5005, 6724, 1343, 649, 6572, 4595, 3653, 2827, 2071,
               1195, 2187, 5894, 4405, 3487, 4746, 2774, 439, 4251, 4067, 2950, 5138, 901, 5708, 190, 993, 6981, 5343,
               6639, 2845, 5840, 4324, 3853, 6129, 3856, 6178, 4329, 6799, 2515, 2716, 6249, 6571, 3059, 3855, 1049,
               6414, 5143, 458, 4450, 6674, 2184, 3240, 2402, 4589, 5375, 5420, 5614, 5347, 1209, 1042, 5118, 1558,
               6730, 2847, 1351, 2503, 3517, 5963, 4611, 3800, 575, 5743, 4704, 3646, 6352, 4620, 5196, 3394, 6385, 314,
               146, 6668, 6467, 5195, 2560, 1671, 1604, 5225, 5973, 1582, 4936, 3707, 2750, 2427, 2381, 5563, 6220,
               2842, 4241, 3549, 2919, 2394, 5017, 2288, 6299, 1865, 5093, 6394, 4148, 692, 3889, 2242, 546, 4696, 225,
               1150, 5242, 1149, 3531, 471, 2013, 6141, 3121, 6848, 1023, 4276, 1759, 2562, 6454, 6270, 4791, 1214,
               4063, 2196, 6069, 2459, 3373, 2168, 370, 5056, 4414, 2767, 4536, 4314, 4172, 6956, 6818, 5585, 4260,
               6118, 6522, 2173, 2136, 5784, 5363, 1222, 255, 26, 3975, 4804, 1091, 5612, 2701, 4410, 5601, 3191, 61,
               5074, 1920, 2252, 655, 1584, 2635, 4760, 3050, 2166, 2475, 5275, 5729, 4905, 217, 4371, 968, 2150, 4474,
               1740, 4, 4052, 1953, 4925, 4541, 5227, 2910, 2140, 1832, 2002, 3672, 719, 1587, 4774, 6485, 3319, 3377,
               931, 1826, 5277, 1962, 5117, 4192, 4846, 1259, 210, 1414, 169, 1779, 4303, 4518, 6202, 4503, 2841, 2495,
               4707, 6241, 723, 2374, 3362, 976, 4770, 6983, 6604, 4690, 4312, 3437, 1954, 5948, 768, 2864, 2081, 1526,
               4459, 5260, 684, 4258, 2186, 4177, 3270, 2872, 589, 1546, 3733, 561, 1235, 1297, 4820, 4646, 571, 4472,
               990, 4934, 274, 2890, 3008, 4778, 3181, 4517, 1075, 3275, 3359, 6565, 3571, 6305, 991, 6100, 5498, 1846,
               1184, 6296, 6122, 2968, 6628, 525, 3814, 5686, 4444, 6102, 4010, 4339, 2938, 2586, 2813, 4268, 5206,
               3273, 5865, 1357, 2989, 5414, 3411, 6205, 220, 3452, 6643, 6551, 2660, 5661, 1575, 3795, 2971, 6626, 786,
               2669, 6331, 1238, 1611, 1412, 2001, 5325, 5510, 5429, 2107, 2029, 6971, 2634, 5104, 4404, 6625, 5153,
               3128, 6032, 2951, 4842, 4652, 4225, 3303, 1874, 5411, 4429, 1342, 6457, 5122, 1243, 1657, 601, 1875,
               5126, 3498, 6962, 1404, 919, 4897, 5155, 2207, 259, 3103, 905, 1207, 1622, 597, 969, 2250, 5884, 1899,
               3934, 3614, 46, 78, 1143, 282, 4353, 177, 966, 4862, 6338, 1793, 3256, 4219, 2267, 787, 1896, 5168, 2054,
               1975, 2590, 489, 1829, 1365, 2751, 5100, 6153, 4213, 1441, 5162, 6343, 231, 500, 6307, 2246, 5415, 958,
               2289, 3637, 558, 6433, 916, 5841, 4143, 3520, 6652, 2097, 5337, 6412, 2269, 1444, 926, 5931, 2026, 4902,
               3016, 4480, 514, 1615, 6763, 3330, 3788, 3305, 3159, 695, 3758, 2977, 3419, 5803, 2419, 779, 5960, 1544,
               6855, 1534, 747, 4321, 403, 6002, 2109, 539, 5972, 3845, 3311, 5057, 306, 5247, 4484, 6283, 2225, 4105,
               2814, 1646, 2303, 1845, 479, 4252, 1478, 1570, 5520, 5021, 6067, 3601, 6497, 1031, 638, 4547, 6939, 995,
               3315, 3172, 2318, 6856, 2991, 743, 6953, 6979, 5517, 1122, 6494, 6104, 5641, 751, 6513, 1383, 2979, 6166,
               1767, 2049, 3175, 2101, 2700, 681, 2254, 6097, 5388, 6458, 6323, 726, 3084, 524, 5535, 1340, 2937, 3000,
               6473, 1635, 6489, 2221, 1996, 5782, 1912, 3775, 246, 2491, 935, 3721, 1455, 3899, 114, 4726, 6059, 744,
               1307, 582, 2565, 1262, 3283, 2574, 1652, 2494, 6872, 3127, 4197, 4828, 1542, 5101, 5909, 4795, 6619, 89,
               3490, 1948, 2908, 2058, 5019, 1992, 2159, 4546, 5813, 5392, 6272, 1229, 4963, 6787, 3820, 6941, 4238,
               6686, 3678, 6401, 2675, 4505, 3274, 4069, 6491, 5822, 5295, 4806, 2718, 6108, 6442, 1069, 6741, 1527,
               5557, 5735, 6396, 3786, 3325, 854, 4005, 504, 211, 754, 2883, 4755, 3932, 1522, 5485, 640, 6264, 2008,
               4496, 6389, 982, 4117, 4216, 6780, 4209, 4731, 2850, 1736, 6582, 6422, 3996, 1393, 6064, 5358, 6598,
               1797, 2734, 755, 2536, 486, 5587, 2069, 5191, 4681, 1449, 4160, 5279, 4874, 1552, 2992, 6380, 1857, 3523,
               2707, 5577, 3160, 3425, 1687, 5383, 1926, 6481, 1732, 1933, 3339, 4085, 3148, 4497, 1729, 2942, 4489,
               2040, 469, 6051, 1275, 3126, 6451, 5969, 6526, 2287, 1480, 4986, 133, 1750, 3038, 2170, 1934, 3513, 379,
               5624, 3040, 2911, 2697, 689, 2433, 1433, 171, 1009, 2670, 2603, 6645, 5574, 5523, 4992, 2548, 3681, 4495,
               5761, 6644, 4557, 5638, 1748, 5128, 3627, 2222, 7, 1586, 5839, 4266, 304, 4554, 998, 3108, 5308, 5077,
               270, 1877, 4045, 608, 1692, 1095, 6013, 2144, 1135, 6387, 1620, 554, 4012, 4455, 6088, 5976, 1276, 5494,
               1385, 6859, 3798, 6783, 5114, 1786, 765, 2104, 797, 4064, 6679, 4983, 2507, 5944, 3570, 1081, 4423, 1557,
               1187, 5499, 4122, 3210, 1055, 1292, 3676, 5847, 1277, 6973, 698, 1425, 3204, 5154, 6187, 3364, 251, 6472,
               245, 3600, 2111, 5952, 3565, 4440, 3810, 4764, 2320, 3933, 741, 5102, 706, 1267, 1958, 630, 766, 3491,
               3195, 6441, 3584, 5043, 3203, 3464, 1170, 668, 4794, 3056, 6583, 1227, 5044, 5864, 4486, 219, 4091, 942,
               1494, 3429, 1678, 5771, 5529, 1086, 2074, 321, 4463, 0, 2731, 6699, 6288, 2510, 6599, 6558, 6627, 293,
               2456, 1817, 2442, 4226, 6938, 4513, 3082, 2417, 2940, 4623, 1029, 5814, 5167, 871, 3738, 4805, 4756,
               4594, 5157, 5698, 3515, 5933, 3090, 5932, 1460, 383, 192, 3472, 4182, 6613, 5701, 4706, 6868, 3674, 14,
               6310, 2508, 2772, 3879, 6126, 1585, 3163, 1873, 1801, 3854, 2825, 2324, 2203, 2060, 96, 4703, 1631, 4847,
               1438, 848, 3644, 4848, 6820, 1515, 2698, 5257, 4735, 1318, 737, 4477, 3986, 3187, 4293, 2204, 2037, 5558,
               4336, 4127, 915, 3941, 477, 6501, 1427, 2291, 2230, 3454, 3083, 622, 3420, 6496, 6336, 2849, 5753, 5866,
               605, 6124, 4797, 1682, 1470, 2654, 69, 6680, 2884, 2913, 3756, 4232, 6488, 6715, 4640, 3024, 4017, 3094,
               2661, 4826, 2, 1755, 5179, 101, 4666, 2279, 3147, 3892, 6591, 2367, 4948, 5248, 5762, 4153, 1667, 3445,
               6826, 2365, 1760, 4814, 3723, 212, 2618, 3974, 3859, 5678, 4308, 6395, 4223, 1858, 3790, 3018, 4663,
               2406, 4529, 3505, 6665, 4601, 2352, 97, 2476, 6226, 2817, 3829, 5469, 5874, 258, 1577, 2048, 4245, 6884,
               6066, 4438, 3026, 1718, 2747, 2683, 5302, 6964, 1120, 1329, 2122, 4351, 6404, 1866, 5441, 5033, 3656,
               2183, 1649, 6523, 1567, 149, 4910, 1960, 3086, 5070, 3750, 6007, 3336, 5561, 2928, 840, 2161, 4577, 5863,
               2017, 777, 3639, 1608, 2211, 1372, 3489, 704, 2564, 887, 2383, 3840, 1221, 1991, 2415, 6776, 6998, 5058,
               4784, 3405, 3885, 1853, 173, 1431, 833, 4890, 4909, 1703, 1901, 1524, 2307, 580, 2112, 2157, 1599, 3370,
               4150, 4604, 1287, 5804, 2648, 6025, 6823, 5053, 2808, 3557, 2554, 5273, 6588, 3698, 3708, 2064, 5353,
               123, 3102, 650, 3766, 1328, 5674, 6707, 6070, 3564, 1776, 1928, 4637, 1702, 1377, 2708, 2853, 1668, 5785,
               6600, 1360, 1300, 5389, 1694, 1050, 6358, 2096, 3819, 5752, 2457, 2043, 3085, 1041, 5476, 3055, 5214,
               6841, 2379, 1528, 2668, 4272, 1803, 1998, 2205, 1916, 680, 3922, 2484, 5165, 5640, 5386, 5770, 3077,
               2105, 3901, 4207, 2481, 2414, 3252, 6160, 4884, 1686, 4688, 5177, 5823, 423, 4026, 5990, 6094, 4980,
               6364, 4629, 5490, 3915, 3184, 2600, 2370, 4210, 4014, 3080, 2996, 386, 2521, 5649, 2625, 4320, 1462,
               4808, 6825, 65, 2359, 2926, 5575, 6294, 2364, 5845, 3469, 2099, 5685, 1696, 1774, 2904, 5400, 4270, 5460,
               3263, 3020, 3912, 5261, 6110, 1854, 5659, 5291, 6177, 446, 769, 4409, 6259, 2705, 4728, 696, 506, 5516,
               2130, 6804, 4768, 6434, 3917, 4953, 5629, 3470, 5905, 1082, 5037, 1074, 4242, 6436, 399, 2869, 6800,
               1956, 4311, 6729, 947, 5426, 2810, 1131, 2119, 3746, 1592, 6885, 5911, 1021, 3104, 1852, 569, 5598, 5737,
               5744, 6514, 5860, 3509, 2077, 3145, 4871, 2023, 5372, 1610, 4381, 456, 6487, 6161, 6864, 1399, 2797, 651,
               1507, 1663, 3025, 528, 3842, 5008, 6431, 6612, 4363, 2876, 2452, 3577, 3811, 685, 3463, 5846, 2436, 560,
               2446, 350, 6324, 3910, 4256, 2829, 5851, 5486, 3238, 4627, 449, 4810, 6508, 5994, 4504, 1994, 6207, 3712,
               3013, 5284, 3596, 697, 2556, 3778, 1397, 6065, 3689, 324, 5552, 4509, 4723, 2612, 4775, 2298, 406, 6351,
               3826, 6189, 977, 4008, 6948, 1145, 5595, 6421, 3031, 636, 6888, 289, 4798, 2742, 4570, 2621, 6150, 5855,
               6566, 5344, 5173, 4344, 3683, 2682, 1198, 4119, 2447, 6142, 2278, 979, 1258, 3379, 5268, 4147, 1923,
               6022, 6290, 6356, 2859, 1070, 1531, 1816, 6879, 2000, 3057, 6388, 4090, 4682, 5209, 4559, 5778, 4889,
               142, 6416, 1485, 1047, 2721, 2584, 5108, 4309, 2089, 6209, 3942, 4398, 2192, 4188, 2084, 2420, 5346,
               1154, 5461, 6918, 1193, 5161, 1909, 4103, 2195, 3312, 1245, 4481, 3957, 559, 1971, 5230, 4018, 6026,
               5309, 2281, 5572, 2656, 891, 3366, 5756, 4827, 1660, 789, 2300, 1285, 5491, 6068, 2327, 2724, 1763, 3406,
               518, 5644, 2113, 545, 6966, 6623, 4096, 77, 6656, 427, 635, 5758, 2691, 6484, 278, 6159, 717, 483, 4837,
               5671, 3436, 5630, 5907, 5357, 2362, 2982, 1493, 6688, 4698, 4465, 3628, 5428, 1574, 1884, 6332, 6525,
               5004, 1945, 4441, 5856, 1855, 5742, 3610, 266, 6012, 4694, 6490, 4265, 3794, 6366, 5147, 472, 5657, 3857,
               5119, 6060, 115, 749, 5633, 4284, 3088, 6384, 2782, 1730, 5350, 4359, 6659, 6204, 6919, 189, 1932, 5495,
               1939, 629, 343, 1828, 701, 3141, 3215, 1871, 5760, 2962, 2496, 6277, 5039, 5604, 2129, 4515, 5611, 6172,
               6469, 3477, 5734, 6692, 2400, 2995, 3715, 3135, 200, 1421, 5080, 5597, 239, 4568, 280, 6258, 6533, 6289,
               2795, 4714, 3699, 6982, 1867, 3100, 408, 4000, 5061, 6242, 4131, 6217, 3039, 6417, 4319, 2237, 3580,
               6553, 5185, 3434, 31, 1695, 3233, 3711, 6718, 4178, 502, 2875, 3431, 5807, 5132, 6811, 2274, 144, 2822,
               6765, 4587, 6218, 6440, 3745, 6713, 3871, 2640, 3742, 3556, 1435, 1172, 2748, 1569, 6829, 825, 1098, 55,
               3924, 3295, 3192, 6943, 1847, 1617, 4246, 3217, 1043, 3688, 51, 3211, 3583, 5997, 4098, 3690, 2474, 2934,
               329, 5218, 5902, 1496, 745, 6376, 1985, 4144, 4485, 5152, 1911, 2086, 4835, 5269, 1448, 2042, 2933, 4415,
               6390, 2493, 1148, 1099, 1278, 6240, 1547, 4721, 1918, 4839, 1398, 6339, 6775, 5282, 662, 1228, 3462,
               5084, 6080, 4767, 1316, 2867, 4562, 2579, 801, 5716, 1284, 4342, 2358, 5012, 174, 5258, 4940, 1717, 3248,
               6618, 4732, 3925, 4984, 6471, 5869, 112, 5569, 3687, 824, 599, 5524, 5038, 3947, 1773, 5449, 2057, 4880,
               6546, 2650, 6506, 1879, 2582, 6573, 378, 2631, 2098, 888, 3267, 2067, 6789, 3067, 2120, 4317, 5793, 2816,
               3028, 224, 4257, 4135, 3545, 5988, 4343, 214, 6052, 1743, 746, 5145, 562, 4945, 3511, 1949, 2999, 5927,
               2483, 3512, 2553, 4205, 4833, 2604, 3629, 6781, 936, 4304, 5211, 583, 537, 5900, 1723, 3704, 4436, 3748,
               3905, 5821, 6958, 2232, 387, 3597, 3834, 2900, 3015, 424, 1102, 6793, 564, 4551, 955, 3439, 5341, 5898,
               1472, 3294, 5064, 4326, 945, 4988, 6590, 5450, 3900, 6127, 3590, 2270, 3228, 3, 6568, 5827, 3397, 4356,
               3402, 6561, 5876, 4998, 5280, 6969, 3112, 5799, 767, 1823, 2542, 519, 21, 1032, 3027, 6703, 778, 4597,
               4171, 3169, 665, 3685, 3224, 577, 3623, 1770, 6271, 5858, 5551, 5549, 4991, 3718, 1108, 3006, 269, 5306,
               3120, 1860, 3886, 2505, 3497, 438, 2369, 243, 868, 3383, 2423, 6120, 648, 3131, 2375, 834, 461, 4956,
               1291, 3813, 3130, 23, 2729, 1726, 5111, 587, 3542, 2534, 4655, 542, 4390, 1147, 4742, 1662, 1333, 800,
               712, 2696, 2471, 6511, 3906, 5578, 3702, 5042, 5726, 5354, 4104, 6254, 1553, 4121, 1573, 4943, 5148,
               1742, 2717, 5271, 319, 4111, 3791, 3548, 5959, 5635, 6006, 3243, 980, 4786, 771, 4011, 6670, 1618, 6827,
               4677, 3105, 1689, 4539, 774, 1115, 338, 3555, 5818, 6779, 1168, 3518, 4053, 6996, 1513, 2687, 3042, 5565,
               1200, 1437, 6089, 6917, 5276, 707, 877, 6381, 493, 994, 911, 4861, 6762, 2304, 2868, 3587, 6392, 5891,
               3353, 2793, 6350, 5566, 3156, 2390, 2257, 4974, 5340, 5477, 4220, 3560, 4508, 5511, 6329, 2647, 6721,
               145, 2892, 1319, 3146, 6714, 3478, 2818, 1139, 963, 5603, 6745, 1836, 4849, 6853, 2922, 3367, 2993, 5831,
               1113, 3207, 4248, 700, 3921, 6369, 1664, 53, 44, 1231, 346, 6749, 5702, 6215, 3372, 5069, 4249, 1837,
               1100, 400, 2124, 3866, 4424, 2575, 671, 2003, 1957, 364, 4873, 1843, 4235, 229, 6597, 2393, 1403, 5092,
               109, 1469, 4221, 1895, 6368, 5406, 5719, 705, 4662, 1738, 2801, 811, 2297, 6984, 1166, 4140, 2218, 2344,
               2236, 1413, 3354, 4034, 2309, 5503, 4715, 544, 4569, 1059, 110, 3945, 5912, 657, 3416, 6248, 4116, 644,
               2843, 5642, 1089, 6297, 2889, 5652, 463, 4525, 1713, 2599, 3393, 3796, 5648, 875, 6927, 2959, 3625, 3320,
               5201, 6543, 2917, 2378, 1144, 4582, 732, 4492, 5899, 758, 3903, 1068, 1355, 827, 3021, 5586, 5424, 2027,
               1127, 1637, 4792, 4817, 641, 4771, 3278, 6611, 3955, 1632, 794, 6156, 5432, 2296, 2580, 5297, 2337, 2162,
               1681, 5378, 1906, 4937, 1432, 5246, 3308, 3232, 6378, 1504, 5983, 5149, 377, 967, 2777, 5294, 886, 6892,
               983, 6040, 5238, 2761, 2543, 3344, 297, 3508, 612, 2121, 1927, 1831, 4660, 6418, 4419, 6222, 464, 1532,
               6945, 5896, 4944, 4406, 4901, 35, 3816, 6460, 1216, 693, 4708, 3868, 1980, 250, 2863, 413, 6723, 844,
               3659, 972, 6184, 5314, 4411, 3073, 3259, 5384, 1731, 4858, 3651, 4987, 3890, 331, 6292, 4437, 6766, 3136,
               362, 3991, 3935, 5667, 4036, 3352, 6466, 4567, 3680, 3780, 3051, 4751, 5889, 5599, 4413, 1800, 452, 2143,
               5397, 1591, 625, 4285, 4278, 5981, 5775, 162, 5382, 773, 6666, 260, 4072, 4946, 2021, 6084, 3981, 1937,
               805, 4581, 5403, 3914, 1722, 1514, 137, 4823, 902, 6638, 538, 6029, 5689, 2533, 4951, 742, 2025, 5897,
               1005, 2728, 1253, 5290, 5051, 3269, 292, 6758, 5576, 4757, 1844, 6214, 5781, 2623, 3264, 3606, 1910,
               2283, 1341, 2624, 6507, 5338, 1959, 5427, 3426, 5106, 5750, 3403, 750, 3033, 5885, 4174, 1990, 6403,
               5172, 389, 1077, 2110, 4039, 4844, 5156, 4965, 2881, 4348, 5901, 2914, 3239, 268, 4013, 2736, 4719, 6194,
               1869, 2082, 333, 2706, 5171, 4705, 4695, 6843, 4060, 1429, 130, 2949, 6992, 6838, 687, 468, 6677, 3521,
               498, 5187, 2305, 5764, 4856, 6924, 2191, 6360, 417, 4900, 2802, 5110, 1358, 3245, 5342, 6079, 4066, 6245,
               2771, 6900, 6452, 3035, 11, 2462, 986, 3876, 1593, 2787, 4130, 4025, 13, 5063, 3694, 151, 3797, 550,
               1559, 4639, 3096, 5066, 3621, 6631, 4361, 5464, 5916, 4181, 6633, 1653, 6162, 2806, 4964, 4046, 5442,
               2219, 2053, 1486, 5681, 4347, 1155, 6185, 1614, 759, 2529, 1062, 842, 1749, 4602, 2154, 3873, 5465, 5966,
               2410, 581, 4592, 6253, 3165, 2010, 4651, 4877, 6636, 4864, 2233, 5452, 4739, 4487, 3107, 6503, 5626,
               2862, 3951, 1379, 4975, 376, 6037, 2147, 81, 5850, 1589, 6344, 3234, 1597, 1324, 828, 2540, 421, 6062,
               557, 3950, 5097, 1390, 1028, 19, 2674, 6563, 1080, 2785, 6716, 1334, 3617, 5605, 428, 2915, 3832, 6860,
               5647, 2616, 411, 32, 2677, 2975, 5024, 2921, 2559, 4630, 323, 4699, 2730, 5285, 1167, 3635, 3298, 3522,
               213, 2273, 4654, 1298, 2739, 6792, 5095, 6186, 3550, 3802, 54, 1492, 4540, 572, 1362, 3060, 1753, 2492,
               1550, 5243, 4822, 5505, 1605, 2464, 2931, 883, 4499, 3782, 5616, 1999, 5120, 5747, 4102, 3140, 5859,
               2790, 6547, 2643, 772, 2334, 1783, 4923, 6702, 6930, 1382, 3961, 3661, 6430, 6734, 556, 4876, 4004, 5444,
               3386, 4094, 2453, 1103, 5938, 6134, 4379, 6493, 1976, 2193, 6870, 849, 6031, 1684, 6614, 238, 1250, 6933,
               2066, 5811, 241, 4230, 1537, 4995, 4586, 2844, 2056, 2387, 4793, 5515, 1183, 3321, 3607, 5475, 4346,
               6893, 5555, 3166, 2206, 2740, 4841, 3566, 4727, 5178, 2637, 27, 4038, 3150, 3492, 2952, 2690, 5579, 6664,
               4933, 2335, 6044, 5919, 2775, 5194, 3482, 1835, 1011, 2468, 1969, 5705, 5059, 4370, 1840, 4290, 6669,
               1882, 2199, 451, 3499, 4345, 6502, 4667, 300, 6515, 1972, 2139, 3558, 5665, 1104, 5651, 6301, 3458, 8,
               301, 4754, 5142, 3023, 5668, 3133, 247, 3980, 1819, 1744, 890, 4003, 1182, 2172, 1428, 223, 6105, 2957,
               4095, 598, 611, 5293, 2284, 2532, 1735, 4155, 6637, 1053, 3538, 4843, 2472, 3929, 6589, 4176, 3781, 5421,
               5467, 4264, 2585, 1015, 6629, 5683, 1239, 633, 6796, 711, 1128, 6743, 6437, 5317, 1838, 2094, 1658, 1416,
               6835, 501, 3703, 1861, 492, 596, 4491, 2923, 2905, 2985, 2784, 4187, 138, 3180, 6256, 2311, 5439, 6555,
               2973, 2422, 6954, 4408, 6048, 3279, 6101, 716, 3636, 4865, 3309, 6286, 4954, 5419, 2967, 5385, 6737, 530,
               3289, 6279, 5312, 4078, 4716, 6528, 1402, 2769, 2799, 510, 2712, 3459, 2518, 4893, 4661, 88, 4855, 2146,
               6201, 4730, 3717, 2174, 5366, 1913, 1650, 4400, 4803, 6524, 1579, 3784, 1938, 131, 6018, 3290, 5000, 956,
               6587, 328, 3697, 2418, 4340, 369, 5412, 2743, 2045, 6315, 4467, 5917, 6925, 6650, 5550, 187, 4183, 803,
               1902, 6475, 4596, 943, 1296, 3858, 3063, 2715, 889, 547, 5190, 4184, 5226, 3407, 1271, 5240, 760, 4139,
               2290, 3053, 2061, 3958, 2133, 2512, 5526, 5832, 1602, 3277, 1233, 1305, 6532, 3199, 6944, 1764, 4606,
               4912, 4952, 6653, 2632, 128, 5650, 3994, 2704, 4763, 4106, 1196, 2595, 1785, 1601, 1691, 5509, 3843, 371,
               20, 2563, 2264, 6512, 6935, 6947, 3923, 1761, 5202, 4296, 1338, 5768, 154, 3534, 29, 3380, 3554, 3533,
               1716, 4331, 677, 4749, 3665, 4452, 5014, 4553, 5081, 3765, 5436, 6478, 2128, 4659, 1951, 3129, 1247, 95,
               2745, 6785, 2444, 2179, 3938, 5016, 1018, 2031, 2965, 1367, 167, 923, 594, 4250, 6676, 100, 1583, 6912,
               3620, 1966, 918, 3920, 2421, 3510, 4397, 80, 6197, 3658, 344, 1856, 590, 699, 87, 962, 1705, 3185, 1942,
               3005, 3831, 6003, 3594, 5170, 5975, 3752, 1025, 1124, 2445, 3488, 4679, 5824, 6955, 5543, 820, 233, 6386,
               2506, 414, 1034, 920, 4377, 1030, 4471, 5970, 818, 1850, 3144, 5183, 398, 50, 2323, 3230, 2630, 917,
               2430, 1603, 4762, 5447, 2517, 1007, 1134, 5634, 108, 1707, 430, 148, 1915, 1436, 5052, 2738, 1878, 4851,
               2398, 5456, 4418, 3456, 4636, 4625, 2322, 6267, 1178, 1317, 440, 3896, 3801, 1153, 2657, 3916, 1111,
               6317, 132, 6408, 682, 2974, 6550, 3895, 3237, 6557, 1508, 3347, 6078, 4691, 5497, 2823, 4866, 4196, 3212,
               491, 5893, 4396, 5720, 3867, 3097, 4733, 6850, 1693, 5765, 1141, 3743, 6663, 2694, 6410, 86, 394, 6960,
               4330, 4680, 1052, 3400, 748, 1827, 4872, 1881, 360, 4521, 2372, 5015, 193, 734, 5534, 1483, 5835, 1313,
               5982, 3343, 3973, 4644, 434, 2852, 1388, 2277, 6751, 2256, 788, 433, 6920, 166, 4358, 5331, 1197, 1997,
               2087, 567, 5986, 5303, 305, 4338, 804, 1476, 4743, 2499, 2363, 757, 4919, 1903, 3206, 3930, 6689, 381,
               2416, 4821, 157, 3229, 2684, 1806, 4202, 2201, 5704, 5115, 4446, 318, 5504, 1804, 2504, 4619, 2351, 2879,
               6262, 4614, 1768, 6397, 6993, 835, 922, 6834, 667, 4815, 5789, 4185, 985, 2032, 1922, 3817, 5459, 3660,
               4108, 3249, 6936, 5413, 1979, 4829, 6227, 952, 5348, 6682, 3931, 3575, 3760, 2314, 1824, 6470, 1475,
               4262, 4747, 4981, 2958, 678, 2070, 1870, 311, 940, 3877, 94, 357, 3574, 6340, 1085, 2189, 866, 3846,
               4157, 4479, 262, 6757, 3168, 5703, 1384, 1371, 6970, 5800, 4578, 850, 1033, 3134, 5200, 1465, 1533, 401,
               4966, 4297, 4021, 4772, 1303, 2549, 4428, 1516, 3965, 853, 6549, 5072, 181, 4393, 6831, 3486, 4049, 937,
               934, 2488, 1117, 5808, 4675, 4380, 1656, 2662, 929, 5192, 6182, 5468, 2838, 1132, 3833, 4112, 170, 287,
               3388, 201, 1330, 663, 2835, 602, 4671, 3662, 3576, 2180, 6874, 762, 2460, 3313, 1119, 4928, 6302, 3809,
               1406, 4186, 6518, 1252, 4164, 3219, 6216, 5006, 5323, 6684, 2095, 6696, 2882, 6615, 2741, 2651, 3390,
               2759, 4305, 4550, 2449, 1409, 1834, 4203, 5727, 5396, 1440, 5367, 286, 6875, 1142, 756, 5862, 928, 6816,
               393, 1677, 3907, 3999, 941, 6667, 1401, 3830, 5109, 4947, 1566, 2754, 3154, 2260, 6592, 6252, 837, 2235,
               4152, 1802, 1114, 277, 4561, 2763, 6744, 5025, 1210, 812, 4169, 1616, 3304, 6712, 5796, 4054, 16, 6474,
               70, 3898, 2018, 785, 453, 3612, 6995, 6355, 3153, 6882, 5393, 3226, 5731, 6282, 3609, 6708, 1523, 5336,
               6446, 2033, 6229, 140, 5002, 1293, 4382, 4626, 3552, 5193, 3682, 3115, 1408, 5872, 4514, 863, 2878, 5011,
               5622, 4365, 3205, 2893, 2570, 6383, 5035, 6991, 3012, 4807, 3142, 327, 6840, 2326, 4388, 2125, 5873,
               3581, 6439, 6607, 2275, 2676, 5049, 3987, 117, 1260, 136, 5034, 508, 180, 1775, 5926, 39, 4247, 867,
               6761, 6866, 2513, 5945, 579, 425, 6453, 5946, 490, 2903, 3335, 3516, 3504, 5492, 4378, 91, 5621, 4813,
               1772, 3381, 1788, 775, 3741, 553, 3475, 6880, 4960, 6932, 2608, 4498, 494, 4092, 6937, 4159, 5223, 6136,
               5047, 6192, 4118, 6965, 4392, 374, 5319, 3123, 912, 932, 1607, 1636, 6348, 2555, 534, 4470, 158, 420,
               987, 1176, 3582, 2497, 1842, 487, 2487, 4475, 6370, 1784, 570, 5180, 2840, 3167, 3956, 5879, 913, 5176,
               6014, 4136, 1512, 1890, 961, 6921, 3881, 1943, 2644, 4852, 1208, 3640, 1386, 5949, 1701, 4802, 6164,
               3634, 5437, 1151, 1112, 2116, 5251, 3218, 6152, 2530, 3559, 256, 1274, 248, 6123, 6039, 6988, 5777, 348,
               126, 604, 882, 1045, 4362, 609, 4043, 6372, 659, 1241, 6569, 176, 5087, 3413, 5883, 2428, 6972, 2210,
               1315, 1986, 5518, 3670, 5645, 1312, 1468, 2489, 3340, 6498, 549, 3768, 3328, 6464, 4165, 3705, 6681,
               6365, 4564, 2873, 739, 4093, 284, 5766, 6459, 1272, 4711, 1039, 632, 3838, 4335, 3253, 832, 3589, 1345,
               4332, 5355, 4445, 1807, 1442, 4973, 2714, 6701, 5794, 4713, 799, 3527, 2384, 2502, 2439, 4191, 3468,
               1648, 6232, 1181, 654, 3862, 5890, 2528, 4832, 5717, 6660, 4364, 160, 4384, 5463, 6538, 2658, 5618, 3387,
               6020, 1576, 4292, 447, 4780, 418, 5493, 6054, 2547, 3068, 2075, 2837, 1376, 6208, 3888, 4195, 1630, 4782,
               2382, 5479, 4894, 5445, 6773, 3337, 5617, 5608, 3603, 1265, 5533, 4212, 4168, 4616, 1961, 2376, 6228,
               4970, 3944, 4903, 5955, 512, 6896, 2686, 5207, 1107, 6211, 1364, 3457, 566, 914, 4931, 2015, 2349, 1554,
               5272, 5834, 4712, 3037, 3977, 4549, 4575, 3036, 6480, 5732, 588, 3762, 194, 6922, 3461, 709, 4664, 4686,
               2247, 4962, 1675, 783, 1798, 4461, 2939, 2752, 1669, 3902, 6106, 1734, 264, 4645, 6846, 4785, 5882, 5189,
               4306, 1251, 407, 6728, 2988, 2588, 4422, 6916, 899, 2614, 6914, 6109, 3446, 4545, 178, 4845, 6055, 2943,
               3713, 5540, 683, 3479, 3417, 1088, 3443, 3066, 2165, 3719, 574, 1477, 802, 2404, 5838, 3361, 1481, 2935,
               6125, 5131, 3052, 215, 30, 548, 1140, 4761, 4023, 2022, 2038, 2399, 4279, 3276, 1024, 1970, 679, 2044,
               1066, 631, 5236, 2214, 6257, 2918, 156, 3174, 4432, 116, 2486, 516, 5679, 6409, 898, 2641, 3847, 1186,
               5870, 1201, 5369, 1126, 1733, 5871, 3828, 4403, 409, 18, 3011, 337, 2356, 5324, 3262, 67, 4375, 4935,
               1335, 1308, 102, 6881, 2987, 3351, 4834, 2158, 5724, 1851, 5402, 2239, 2455, 2832, 4218, 6940, 713, 5151,
               3962, 6260, 3254, 391, 829, 5470, 3392, 317, 1152, 106, 1204, 3852, 3323, 4146, 3047, 3774, 3968, 578,
               3928, 1619, 1175, 3747, 2605, 5537, 2443, 191, 5407, 6842, 1013, 37, 910, 315, 5937, 3960, 432, 906,
               1246, 4286, 4669, 5451, 1452, 3714, 3734, 6891, 4163, 1002, 5792, 2065, 5923, 4799, 4137, 5096, 1905,
               5654, 4442, 3058, 3480, 6675, 3785, 4227, 234, 3732, 4511, 6171, 2576, 6074, 272, 2271, 1964, 5880, 5787,
               1530, 4544, 4961, 2779, 3396, 1344, 5581, 5733, 6170, 294, 6200, 4507, 565, 3602, 5409, 2177, 3638, 729,
               5539, 1003, 5843, 4224, 5692, 603, 903, 6427, 3807, 6230, 2132, 2151, 5351, 470, 6541, 600, 1370, 4906,
               6747, 6047, 4133, 3864, 6584, 5186, 2403, 688, 1935, 3875, 299, 2208, 6322, 5852, 2448, 5030, 2768, 4868,
               5360, 2786, 422, 3803, 1974, 5404, 1347, 6083, 4086, 6049, 5266, 5250, 6199, 6000, 4687, 60, 4576, 4074,
               2693, 2212, 4752, 876, 5602, 3650, 5546, 6179, 3202, 4201, 925, 6544, 5795, 4024, 1125, 6595, 924, 3767,
               4028, 1511, 127, 22, 3631, 2685, 1137, 5079, 3501, 2749, 4612, 5430, 2781, 98, 6001, 3266, 232, 2350,
               753, 2607, 5307, 5710, 1387, 3773, 25, 4548, 308, 4918, 1568, 1670, 637, 6374, 5222, 540, 3642, 6212,
               686, 1361, 1708, 2263, 2079, 6576, 981, 3331, 1924, 5085, 6617, 5296, 884, 1206, 1061, 6725, 1158, 3423,
               2131, 3844, 5379, 1273, 5339, 933, 2628, 1264, 5935, 527, 2703, 353, 1381, 1519, 6219, 2961, 3588, 1087,
               5662, 2617, 5099, 2127, 168, 3032, 488, 5833, 5993, 4051, 3214, 5816, 1813, 3783, 3959, 6527, 2114, 816,
               6038, 6774, 4938, 1266, 606, 1160, 15, 6050, 4750, 568, 3334, 495, 5763, 6863, 2413, 2100, 6420, 4041,
               296, 410, 5950, 1289, 2870, 1509, 6828, 147, 878, 3125, 396, 2241, 5580, 3539, 5487, 1639, 313, 6719,
               2216, 3318, 6087, 715, 1950, 6482, 2134, 5532, 1952, 2329, 2028, 4777, 1311, 2477, 1304, 6678, 2798,
               6839, 4473, 1685, 5829, 3967, 4443, 316, 6250, 3891, 5335, 3158, 6244, 806, 1064, 3827, 2744, 4281, 2828,
               5798, 1633, 5327, 5521, 6167, 796, 6435, 543, 860, 1188, 6268, 2524, 2059, 497, 5589, 2760, 4433, 728,
               4253, 5356, 6564, 416, 3242, 984, 5915, 5556, 676, 6736, 2811, 1683, 5395, 999, 6278, 1356, 4867, 5263,
               437, 6883, 6732, 4299, 4725, 165, 3201, 2645, 6206, 6540, 6602, 2998, 3740, 6112, 1302, 496, 859, 4722,
               1757, 3171, 5496, 6093, 3541, 973, 6113, 1898, 2557, 4449, 84, 2340, 2295, 2638, 2531, 6810, 6622, 2265,
               457, 1310, 2976, 703, 731, 288, 938, 5675, 3626, 5913, 2591, 1638, 2280, 6158, 2276, 1391, 1434, 5462,
               5655, 5830, 3176, 184, 6461, 2102, 1572, 2960, 3572, 3045, 34, 1790, 4641, 1688, 5292, 593, 56, 345,
               6801, 1418, 3043, 4294, 2465, 4179, 3641, 1463, 1697, 4125, 6978, 5448, 5326, 2980, 6949, 1447, 6987,
               5023, 4142, 5920, 6951, 2231, 1256, 1811, 4700, 3299, 6997, 3355, 3349, 2812, 3061, 1439, 2243, 1116,
               6952, 3221, 1833, 6231, 3918, 1242, 3841, 2753, 563, 2611, 152, 5745, 6379, 1600, 57, 1138, 325, 3250,
               5199, 1174, 2511, 2597, 2137, 6654, 4457, 3432, 3409, 1396, 139, 5374, 2941, 6807, 2858, 4598, 6213,
               1623, 6529, 6005, 129, 6091, 781, 6803, 6361, 1288, 5759, 4875, 6071, 5573, 3410, 5583, 155, 5009, 3792,
               5995, 5987, 5682, 467, 4395, 2569, 1352, 1279, 2202, 5408, 2610, 5318, 6578, 5036, 3235, 4113, 2138,
               4853, 2395, 5220, 6147, 4310, 3081, 4134, 620, 5022, 4138, 823, 4283, 4982, 205, 6306, 6004, 5571, 429,
               3649, 2321, 436, 198, 5562, 1146, 4932, 4574, 5394, 485, 1868, 484, 5677, 6929, 1612, 3293, 5943, 5249,
               5359, 5817, 4180, 6813, 1426, 3329, 3971, 6135, 2479, 3101, 4523, 763, 2004, 1325, 642, 6963, 1745, 817,
               4259, 5593, 3667, 2255, 3563, 1495, 93, 6536, 6649, 595, 2887, 3287, 290, 2986, 6042, 6531, 1548, 2720,
               2886, 2719, 3152, 2953, 150, 2851, 944, 3544, 4280, 4020, 257, 2175, 6567, 227, 3149, 1163, 591, 5473,
               5696, 843, 3139, 3324, 5721, 6154, 1765, 4502, 3502, 1709, 6620, 1254, 971, 1374, 1331, 4642, 5380, 3818,
               6999, 2385, 6581, 5825, 1984, 761, 448, 2568, 4670, 480, 4869, 5886, 1459, 3093, 4831, 4579, 4634, 1756,
               6975, 1281, 6655, 334, 354, 4558, 2282, 6224, 6321, 3003, 5169, 4298, 1777, 2391, 1727, 4668, 2182, 2846,
               4145, 6132, 6116, 847, 3196, 412, 1737, 275, 1261, 4334, 5286, 2860, 4464, 6144, 6221, 1165, 3116, 3848,
               5313, 1978, 236, 3200, 4368, 4302, 673, 455, 3997, 3435, 6603, 2030, 1624, 2088, 3257, 4979, 5046, 3919,
               1595, 2051, 92, 2972, 5928, 2411, 2932, 6349, 2377, 4955, 1020, 3693, 5032, 643, 2854, 6499, 1280, 2047,
               6986, 3261, 5281, 3317, 6819, 73, 3079, 2800, 2123, 791, 4044, 397, 6704, 6119, 460, 2167, 3368, 1497,
               645, 349, 466, 5434, 1046, 5310, 874, 2830, 4939, 4812, 4483, 6077, 3926, 1894, 634, 6769, 6889, 76,
               1191, 2213, 6748, 3111, 6821, 2780, 2550, 1038, 3316, 3385, 1766, 822, 2437, 2115, 6353, 2927, 782, 1400,
               1977, 5007, 5045, 1263, 4608, 909, 894, 2713, 459, 3812, 5967, 1136, 2877, 5458, 1457, 235, 12, 672,
               1521, 3604, 5265, 5159, 4123, 4643, 4538, 5940, 6236, 474, 3726, 1931, 6265, 6415, 6011, 5418, 6198,
               4552, 1714, 1105, 885, 320, 6411, 1424, 1699, 2764, 5687, 3966, 6057, 4913, 1629, 3936, 6115, 395, 3408,
               442, 3227, 2345, 4357, 5857, 175, 4885, 780, 3481, 4391, 6862, 443, 2266, 2527, 3029, 1821, 2766, 5978,
               90, 5836, 5416, 1157, 4451, 3850, 1019, 1848, 526, 6432, 1240, 2155, 392, 551, 1326, 1904, 2839, 6505,
               4263, 2558, 4887, 6691, 5332, 6255, 2498, 4942, 2331, 6764, 4476, 2789, 6907, 6904, 5322, 2249, 6438,
               3430, 368, 3988, 120, 1549, 3297, 6148, 4420, 870, 4555, 6931, 2108, 1782, 445, 5010, 1482, 3418, 2336,
               4048, 3954, 4354, 2389, 119, 3002, 1423, 5646, 6710, 4929, 4037, 4274, 2539, 4891, 6928, 3371, 6073, 351,
               592, 6419, 6425, 4254, 2016, 4830, 5304, 6980, 4243, 2583, 2478, 6287, 3095, 4007, 6477, 6630, 2106,
               3220, 2897, 265, 584, 1642, 3300, 4583, 3729, 4478, 5088, 4649, 2526, 5219, 4236, 4217, 6019, 4787, 2142,
               5320, 3485, 4522, 1680, 2663, 3484, 141, 5875, 3982, 4693, 1405, 3404, 6946, 3878, 3525, 6673, 6791,
               5697, 5090, 5130, 6010, 4128, 3346, 626, 3632, 2020, 1268, 2649, 1368, 312, 4015, 724, 2581, 1830, 3236,
               3622, 5141, 1453, 2006, 5591, 6601, 5094, 5050, 2401, 6318, 6746, 3271, 5971, 3474, 4633, 4372, 3579,
               5377, 2710, 310, 6444, 6117, 4124, 4047, 3503, 4506, 2432, 2245, 4857, 3805, 5091, 996, 4407, 5968, 2659,
               2834, 1643, 499, 4710, 4394, 1219, 6280, 4528, 3989, 5786, 382, 6035, 6028, 5082, 363, 75, 964, 4059,
               2238, 5127, 3669, 3401, 3132, 4591, 1389, 4930, 6085, 5144, 5769, 5711, 1056, 708, 4926, 907, 1454, 6276,
               2735, 124, 3836, 2602, 4198, 3216, 5592, 2819, 5805, 4401, 3030, 5636, 6621, 74, 2135, 5425, 752, 99,
               3117, 197, 6520, 5345, 5184, 1337, 6858, 5774, 6413, 3143, 6989, 1596, 1456, 770, 2794, 4737, 628, 2490,
               5653, 3310, 6735, 2978, 5895, 85, 1561, 5609, 6606, 1180, 1825, 230, 4141, 5438, 4950, 6045, 3543, 1295,
               3725, 1598, 3675, 3265, 3046, 5488, 6016, 4759, 5446, 1001, 5718, 1078, 2727, 2681, 4057, 6806, 3537,
               5150, 5089, 3151, 5113, 6133, 856, 2316, 5906, 4282, 6771, 2538, 4175, 5481, 1921, 2966, 5700, 6239,
               3415, 1525, 4301, 2169, 185, 6562, 1164, 3078, 505, 2523, 3976, 2073, 6193, 2244, 6605, 2990, 1299, 6662,
               6755, 4490, 2473, 3613, 3076, 4035, 1073, 3241, 4702, 2024, 881, 5029, 6334, 3182, 3179, 1249, 1981,
               6693, 1458, 273, 5472, 5741, 5116, 1739, 4533, 79, 2636, 6, 2325, 3616, 2302, 5953, 6448, 5961, 2930,
               3824, 3720, 4002, 6041, 4957, 1121, 1487, 3177, 2343, 5440, 3911, 1715, 3865, 6328, 810, 1839, 6802,
               5502, 3164, 4360, 1930, 6795, 2434, 4082, 6483, 6190, 2546, 4107, 6857, 3684, 2836, 2228, 3285, 3222,
               2571, 5370, 3441, 5232, 5235, 6852, 4800, 2292, 221, 6790, 6311, 6151, 6698, 5067, 4307, 2689, 3753,
               6400, 340, 2407, 3246, 454, 529, 6847, 1217, 1539, 2458, 4896, 3884, 6357, 4809, 6812, 6017, 5797, 5267,
               2091, 5254, 6915, 2642, 1886, 6111, 2090, 1889, 3757, 4673, 6777, 813, 3837, 5376, 4907, 1171, 5903, 865,
               2012, 339, 2227, 3089, 2078, 6923, 6320, 1862, 6768, 475, 2954, 335, 3197, 6641, 3054, 5780, 4519, 1520,
               5454, 1380, 6642, 4228, 3619, 2308, 6449, 1535, 2601, 6056, 5691, 1809, 4878, 195, 2117, 2776, 4603, 997,
               3686, 6130, 3292, 6967, 908, 3883, 4924, 6959, 4512, 6246, 2163, 2906, 5530, 5299, 153, 5060, 656, 3764,
               6778, 3969, 5274, 2145, 5018, 5352, 216, 6174, 5482, 6335, 1888, 2805, 3048, 6492, 879, 207, 2516, 3673,
               2317, 6043, 4349, 3213, 6594, 5746, 873, 6495, 3779, 6157, 6739, 4658, 6647, 5522, 6407, 624, 6281, 1799,
               690, 5740, 2594, 2215, 6586, 6836, 4766, 3701, 2544, 5362, 5506, 4261, 627, 6911, 858, 4161, 3995, 203,
               6950, 242, 6977, 3137, 356, 838, 6086, 6309, 6456, 2627, 710, 2141, 4650, 2509, 3908, 1545, 1501, 3835,
               6304, 252, 2896, 669, 1571, 2441, 509, 6391, 5623, 4288, 1914, 4588, 6428, 3949, 4366, 5922, 4031, 4776,
               1698, 6976, 6596, 5287, 6899, 623, 814, 3473, 6046, 2181, 4030, 3804, 5613, 3493, 6319, 1257, 2567, 653,
               4494, 4200, 795, 226, 1769, 1314, 2240, 1346, 5694, 3448, 6486, 6367, 617, 6519, 4389, 6574, 111, 3374,
               6103, 3551, 6099, 6861, 5105, 5637, 3422, 5435, 5725, 1012, 2737, 6308, 664, 419, 3391, 839, 6363, 5514,
               2757, 5333, 1654, 4077, 3737, 3460, 3530, 1294, 855, 3173, 4083, 2699, 1820, 2519, 6770, 2176, 6733,
               2820, 1336, 271, 3983, 4898, 3569, 1666, 1430, 1190, 2758, 3652, 6720, 3599, 4882, 1058, 3453, 2726, 66,
               2371, 2573, 1588, 4607, 6143, 1036, 6468, 4425, 1796, 1051, 4162, 3645, 3789, 3943, 3561, 4684, 3124,
               1363, 5957, 2695, 6463, 6717, 2945, 405, 1354, 4417, 948, 1540, 957, 6165, 5288, 2438, 4056, 1350, 6575,
               6985, 5826, 2197, 3428, 5417, 5810, 3442, 5224, 1109, 1787, 2821, 2855, 3562, 1349, 2072, 6325, 5828,
               1983, 1417, 2466, 5198, 6709, 4967, 3225, 2907, 1940, 6168, 541, 1419, 5951, 1320, 4376, 6337, 3657,
               5028, 2223, 2361, 2746, 6180, 5210, 5615, 4674, 3087, 2332, 2034, 5538, 1661, 3710, 1536, 4075, 2009,
               6284, 4211, 3369, 3288, 6251, 4584, 4458, 6646, 4657, 6876, 5545, 2680, 3722, 2773, 4081, 5301, 1192,
               4571, 4027, 764, 2520, 6090, 4870, 4233, 6273, 6579, 1673, 5914, 6382, 2268, 555, 1443, 531, 4352, 4318,
               2655, 4215, 3727, 2392, 4524, 2301, 5391, 1017, 6053, 1690, 4385, 3605, 4796, 6683, 1097, 6275, 5256,
               5809, 2083, 1503, 1712, 4914, 2435, 2653, 5867, 384, 5062, 5748, 2874, 3438, 1578, 2293, 6854, 6577,
               6405, 135, 2791, 3251, 3389, 3433, 3696, 6455, 2501, 2902, 291, 4482, 3735, 3952, 4454, 2164, 1484, 3787,
               5316, 3193, 2561, 209, 6510, 4126, 6450, 5631, 5244, 2899, 3643, 3500, 4291, 2639, 735, 5027, 4488, 6886,
               5767, 3679, 1026, 3770, 613, 950, 1947, 5513, 2226, 1232, 2522, 2804, 1225, 2126, 3519, 5466, 3363, 6377,
               372, 4915, 2732, 6957, 2629, 4895, 6844, 2596, 5264, 2366, 4613, 5892, 975, 4295, 6824, 2783, 2171, 3567,
               3754, 4167, 1502, 3618, 809, 5627, 2248, 2826, 2578, 2339, 2866, 6128, 365, 3069, 254, 2909, 1067, 5806,
               1499, 3939, 6081, 1795, 2148, 2609, 5013, 6234, 3869, 2888, 2537, 307, 2386, 5328, 5182, 1805, 1818,
               1213, 3970, 959, 4447, 5525, 5471, 482, 5568, 5457, 3953, 2824, 2592, 3483, 1270, 2994, 1791, 1808, 5158,
               3851, 670, 1094, 1411, 1269, 1665, 4239, 6902, 2046, 5321, 5985, 4435, 6076, 6690, 3110, 2328, 992, 616,
               2306, 4214, 2807, 5474, 4193, 1710, 6009, 4383, 2454, 2262, 4042, 3880, 5791, 341, 2500, 862, 5501, 4840,
               2426, 6316, 4922, 666, 2646, 431, 4100, 5941, 5048, 2587, 1704, 1529, 6398, 3382, 989, 1290, 6760, 6138,
               5216, 6291, 5228, 6095, 2885, 6640, 2080, 4089, 6479, 5315, 1955, 159, 4032, 6903, 1841, 3978, 5422,
               3465, 6809, 4724, 5055, 5779, 5939, 675, 5643, 1781, 5670, 5989, 4240, 4426, 6155, 2525, 2341, 1118,
               3280, 2626, 4689, 1322, 798, 5112, 3849, 5980, 3598, 2891, 3666, 1093, 1244, 2606, 3072, 3268, 4734,
               6300, 4773, 4132, 298, 4501, 4718, 4917, 2552, 208, 6878, 5606, 6333, 3049, 6235, 2294, 1810, 6685, 5215,
               36, 4672, 4605, 1500, 1283, 1556, 2619, 4231, 1060, 49, 652, 3529, 3771, 5197, 5136, 3444, 6909, 4322,
               1543, 6371, 2944, 5619, 4156, 1202, 4836, 4692, 3494, 4158, 2041, 5582, 1491, 3964, 253, 4720, 6833, 869,
               5837, 1451, 4783, 5298, 6072, 163, 3350, 1551, 1506, 4989, 5212, 4665, 3091, 6137, 2796, 1490, 4997,
               2577, 2983, 3209, 1872, 4412, 515, 846, 2153, 2149, 3162, 182, 5221, 1907, 5590, 852, 522, 3808, 2665,
               28, 4838, 2920, 1410, 4469, 517, 6797, 3307, 366, 1353, 6632, 1876, 5600, 380, 6651, 5570, 2467, 5484,
               2469, 6926, 326, 694, 2092, 1301, 1084, 1016, 6191, 2178, 2615, 3302, 1936, 2762, 3189, 6537, 2299, 1917,
               2916, 6034, 2063, 3022, 535, 3099, 2545, 5878, 5068, 4788, 1674, 2912, 5103, 646, 404, 5203, 6671, 954,
               6871, 2461, 4367, 5241, 5237, 1963, 523, 5658, 1101, 3338, 4534, 3546, 5910, 4009, 4881, 5596, 4273,
               2929, 45, 5924, 6869, 1754, 4350, 5673, 6476, 3291, 1473, 1407, 1161, 1815, 4016, 5098, 3190, 4033, 3887,
               4717, 6181, 1420, 6542, 5554, 3255, 5812, 1130, 3593, 1849, 6462, 3455, 691, 5208, 4277, 105, 6346, 718,
               1789, 615, 3424, 5888, 5588, 10, 4531, 1679, 4920, 3188, 3398, 2833, 6738]
    print "Random permutation :) "
    datasets[0] = datasets[0][np.array(indices)]  # get permutation
    datasets[6] = datasets[6][np.array(indices)]
    perf = train_conv_net(datasets,
                          U,  # W2V matrix
                          word_idx_map,
                          img_w=W.shape[1],
                          lr_decay=0.95,
                          filter_hs=[3, 4, 5],
                          conv_non_linear="relu",
                          hidden_units=[100, 2],  # hidden_units=[100,2]
                          shuffle_batch=True,
                          n_epochs=20,  # 20
                          sqr_norm_lim=9,
                          non_static=non_static,
                          batch_size=50,
                          dropout_rate=[0.4],
                          pi_params=[0, 0],
                          C=6.,
                          patience=5)  # 20
    q_results.append(perf[0])
    p_results.append(perf[1])
    print 'teacher network q: ', str(np.mean(q_results))
    print 'studnet network p: ', str(np.mean(p_results))
