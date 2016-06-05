from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

import lasagne

from . import trainer, dnn_objective




class AbstractPixelClassifier (object):
    @classmethod
    def for_model(cls, network_build_fn, params_path=None, *args, **kwargs):
        """
        Construct a classifier, given a network building function
        and an optional path from which to load parameters.
        :param network_build_fn: network builder function of the form `fn(input_var, **kwargs) -> lasagne_layer`
        that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
        :param params_path: [optional] path from which to load network parameters
        :return: a classifier instance
        """
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        # Build the network
        print("Building model and compiling functions...")
        network = network_build_fn(input_var=input_var, **kwargs)
        # If a parameters path is provided, load them
        if params_path is not None:
            trainer.load_model(params_path, network)
        return cls(input_var, target_var, network, *args, **kwargs)

    @classmethod
    def for_network(cls, network, *args, **kwargs):
        """
        Construct a classifier instance, given a pre-built network.
        :param network: pre-built network, in the form of a Lasagne layer
        :param args:
        :param kwargs:
        :return:
        """
        # Construct
        input_var = network.get_network_input_var(network)
        target_var = T.ivector('targets')
        return cls(input_var, target_var, network, *args, **kwargs)


def _flatten_pixels_lasagne_layer(layer):
    # (S,C,H,W) -> (C,S,H,W)
    layer = lasagne.layers.DimshuffleLayer(layer, (1, 0, 2, 3))
    # (C,S,H,W) -> (C,SHW)
    layer = lasagne.layers.FlattenLayer(layer, outdim=2)
    # (C,SHW) -> (SHW,C)
    layer = lasagne.layers.DimshuffleLayer(layer, (1, 0))
    return layer

def _unflatten_pixels(x, block_shape):
    # (SHW,C) -> (C,SHW)
    x = np.rollaxis(x, 1, 0)
    # (C,SHW) -> (C,S,H,W)
    x = x.reshape((x.shape[0], -1) + block_shape)
    # (C,S,H,W) -> (S,C,H,W)
    x = np.rollaxis(x, 1, 0)
    return x



class ImagePixelClassifier (AbstractPixelClassifier):
    def __init__(self, input_var, target_var, final_layer, updates_fn=None):
        """
        Constructor - construct an `ImageClassifier` instance given variables for
        input, target and a final layer (a Lasagne layer)
        :param input_var: input variable, a Theano variable
        :param target_var: target variable, a Theano variable
        :param final_layer: final layer, a Lasagne layer
        :param updates_fn: [optional] a function of the form `fn(cost, params) -> updates` that
            generates update expressions given the cost and the parameters to update using
            an optimisation technique e.g. Nesterov Momentum:
            `lambda cost, params: lasagne.updates.nesterov_momentum(cost, params,
                learning_rate=0.002, momentum=0.9)`
        """
        self.input_var = input_var
        self.target_var = target_var
        self.final_layer = final_layer
        self.softmax = dnn_objective.TemperatureSoftmax()

        network = lasagne.layers.NonlinearityLayer(_flatten_pixels_lasagne_layer(final_layer), self.softmax)
        self.network = network

        # TRAINING

        # Get an expression representing the network's output
        prediction = lasagne.layers.get_output(network)
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        if updates_fn is None:
            updates = lasagne.updates.nesterov_momentum(
                    loss.mean(), params, learning_rate=0.01, momentum=0.9)
        else:
            updates = updates_fn(loss.mean(), params)

        # EVALUATION - VALIDATION, TEST, PREDICTION

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_prediction = lasagne.layers.get_output(network, deterministic=True)
        # Create evaluation loss expression
        eval_loss = lasagne.objectives.categorical_crossentropy(eval_prediction,
                                                                target_var)
        # Create an expression for error count
        test_err = T.sum(T.neq(T.argmax(eval_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        t_train_fn = theano.function([input_var, target_var], loss.sum(), updates=updates)
        def train_fn(x, y):
            return t_train_fn(x, y.flatten())
        self._train_fn = train_fn

        # Compile a function computing the validation loss and error:
        t_val_fn = theano.function([input_var, target_var], [eval_loss.sum(), test_err])
        def val_fn(x, y):
            return t_val_fn(x, y.flatten())
        self._val_fn = val_fn

        # Compile a function computing the predicted probability
        self._predict_prob_fn = theano.function([input_var], eval_prediction)


        # Construct a trainer
        self.trainer = trainer.Trainer()
        # Provide with training function
        self.trainer.train_with(train_batch_fn=self._train_fn,
                                train_epoch_results_check_fn=self._check_train_epoch_results)
        # Evaluate with evaluation function, the second output value - error rate - is used for scoring
        self.trainer.evaluate_with(eval_batch_fn=self._val_fn, validation_improved_fn=1)
        # Set the epoch logging function
        self.trainer.report(epoch_log_fn=self._epoch_log)
        # Tell the trainer to store parameters when the validation score (error rate) is best
        # self.trainer.retain_best_scoring_state_of_updates(updates)
        self.trainer.retain_best_scoring_state_of_network(network)


    def _check_train_epoch_results(self, epoch, train_epoch_results):
        if np.isnan(train_epoch_results).any():
            return 'Training loss of NaN'
        else:
            return None


    def _epoch_log(self, epoch_number, delta_time, train_results, val_results, test_results):
        """
        Epoch logging callback, passed to the `self.trainer.report()`
        """
        items = []
        items.append('Epoch {}/{} took {:.2f}s'.format(epoch_number + 1, self.trainer.num_epochs, delta_time))
        items.append('train loss={:.6f}'.format(train_results[0]))
        if val_results is not None:
            items.append('val loss={:.6f}, val err={:.2%}'.format(val_results[0], val_results[1]))
        if test_results is not None:
            items.append('test err={:.2%}'.format(test_results[1]))
        return ', '.join(items)


    def predict_prob(self, X, y_shape, batchsize=500, temperature=None, batch_xform_fn=None):
        """
        Predict probabilities for input samples
        :param X: input samples
        :param batchsize: [optional] mini-batch size default=500
        :param temperature: [optional] softmax temperature
        :return:
        """
        y = []
        if temperature is not None:
            self.softmax.temperature = temperature
        for batch in self.trainer.batch_iterator(X, batchsize=batchsize, shuffle=False):
            if batch_xform_fn is not None:
                batch = batch_xform_fn(batch)
            y_batch = self._predict_prob_fn(batch[0])
            y_batch = _unflatten_pixels(y_batch, y_shape)
            y.append(y_batch)
        y = np.concatenate(y, axis=0)
        if temperature is not None:
            self.softmax.temperature = 1.0
        return y


    def predict_cls(self, X, y_shape, batchsize=500, batch_xform_fn=None):
        prob = self.predict_prob(X, y_shape, batchsize=batchsize, batch_xform_fn=batch_xform_fn)
        return np.argmax(prob, axis=1)
