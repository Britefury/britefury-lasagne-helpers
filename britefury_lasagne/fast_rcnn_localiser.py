from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

import lasagne

from . import trainer


class AbstractFastRCNNLocaliser (object):
    @classmethod
    def for_model(cls, network_build_fn, n_anchor_boxes, params_path=None, *args, **kwargs):
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
        objectness_var = T.ivector('objectness')
        obj_msk_var = T.vector('obj_mask')
        boxes_var = T.matrix('rel_boxes')
        box_msk_var = T.matrix('box_mask')

        # Build the network
        print("Building model and compiling functions...")
        network = network_build_fn(input_var=input_var, **kwargs)
        # If a parameters path is provided, load them
        if params_path is not None:
            trainer.load_model(params_path, network)
        return cls(input_var, objectness_var, obj_msk_var, boxes_var, box_msk_var, network, n_anchor_boxes, *args, **kwargs)

    @classmethod
    def for_network(cls, network, n_anchor_boxes, *args, **kwargs):
        """
        Construct a classifier instance, given a pre-built network.
        :param network: pre-built network, in the form of a Lasagne layer
        :param args:
        :param kwargs:
        :return:
        """
        # Construct
        input_var = network.get_network_input_var(network)
        objectness_var = T.ivector('objectness')
        obj_msk_var = T.vector('obj_mask')
        boxes_var = T.matrix('rel_boxes')
        box_msk_var = T.matrix('box_mask')
        return cls(input_var, objectness_var, obj_msk_var, boxes_var, box_msk_var, network, n_anchor_boxes, *args, **kwargs)



class ImageFastRCNNLocaliser (AbstractFastRCNNLocaliser):
    def __init__(self, input_var, objectness_var, obj_msk_var, boxes_var, box_msk_var, top_layer,
                 n_anchor_boxes, updates_fn=None):
        """
        Constructor - construct an `ImageClassifier` instance given variables for
        input, target and a final layer (a Lasagne layer)
        :param input_var: input variable, a Theano variable
        :param target_var: target variable, a Theano variable
        :param top_layer: final layer, a Lasagne layer
        :param updates_fn: [optional] a function of the form `fn(cost, params) -> updates` that
            generates update expressions given the cost and the parameters to update using
            an optimisation technique e.g. Nesterov Momentum:
            `lambda cost, params: lasagne.updates.nesterov_momentum(cost, params,
                learning_rate=0.002, momentum=0.9)`
        """
        self.input_var = input_var
        self.objectness_var = objectness_var
        self.obj_msk_var = obj_msk_var
        self.boxes_var = boxes_var
        self.box_msk_var = box_msk_var
        self.top_layer = top_layer
        self.n_anchor_boxes = n_anchor_boxes

        # Objectness classifier
        l_objectness = lasagne.layers.DenseLayer(top_layer, num_units=n_anchor_boxes*2, nonlinearity=None)
        l_objectness = lasagne.layers.ReshapeLayer(l_objectness, (-1, 2))
        l_objectness = lasagne.layers.NonlinearityLayer(l_objectness, nonlinearity=lasagne.nonlinearities.softmax)
        self.l_objectness = l_objectness

        # Relative boxes
        l_rel_boxes = lasagne.layers.DenseLayer(top_layer, num_units=n_anchor_boxes*4, nonlinearity=None)
        l_rel_boxes = lasagne.layers.ReshapeLayer(l_rel_boxes, ([0], n_anchor_boxes, 4))
        self.l_rel_boxes = l_rel_boxes

        self.final_layers = [l_objectness, l_rel_boxes]

        # TRAINING

        pred_obj = lasagne.layers.get_output(l_objectness)
        pred_obj_loss = lasagne.objectives.categorical_crossentropy(pred_obj, self.objectness_var) * self.obj_msk_var

        pred_box = lasagne.layers.get_output(l_rel_boxes)
        pred_box_loss = lasagne.objectives.squared_error(pred_box, self.boxes_var) * self.box_msk_var

        train_loss = pred_obj_loss.mean() + pred_box_loss.mean()


        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params([l_objectness, l_rel_boxes], trainable=True)
        if updates_fn is None:
            updates = lasagne.updates.nesterov_momentum(
                    train_loss, params, learning_rate=0.01, momentum=0.9)
        else:
            updates = updates_fn(train_loss, params)

        # EVALUATION - VALIDATION, TEST, PREDICTION

        eval_pred_obj = lasagne.layers.get_output(l_objectness, deterministic=True)
        eval_pred_obj_loss = lasagne.objectives.categorical_crossentropy(eval_pred_obj, self.objectness_var) * self.obj_msk_var

        eval_pred_box = lasagne.layers.get_output(l_rel_boxes, deterministic=True)
        eval_pred_box_loss = lasagne.objectives.squared_error(eval_pred_box, self.boxes_var) * self.box_msk_var


        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self._train_fn = theano.function([input_var, objectness_var, obj_msk_var, boxes_var, box_msk_var],
                                         [pred_obj_loss.sum(), pred_box_loss.sum()], updates=updates)

        # Compile a function computing the validation loss and error:
        self._val_fn = theano.function([input_var, objectness_var, obj_msk_var, boxes_var, box_msk_var],
                                       [eval_pred_obj_loss.sum() + eval_pred_box_loss.sum(),
                                        eval_pred_obj_loss.sum(), eval_pred_box_loss.sum()])

        # Compile a function computing the predicted probability
        self._predict_prob_fn = theano.function([input_var], [eval_pred_obj, eval_pred_box])


        # Construct a trainer
        self.trainer = trainer.Trainer()
        # Provide with training function
        self.trainer.train_with(train_batch_fn=self._train_fn,
                                train_epoch_results_check_fn=self._check_train_epoch_results)
        # Evaluate with evaluation function, the second output value - error rate - is used for scoring
        self.trainer.evaluate_with(eval_batch_fn=self._val_fn, validation_improved_fn=0)
        # Set the epoch logging function
        self.trainer.report(epoch_log_fn=self._epoch_log)
        # Tell the trainer to store parameters when the validation score (error rate) is best
        # self.trainer.retain_best_scoring_state_of_updates(updates)
        self.trainer.retain_best_scoring_state_of_network(self.final_layers)


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


    def predict_boxes(self, X, batchsize=500, batch_xform_fn=None):
        """
        Predict probabilities for input samples
        :param X: input samples
        :param batchsize: [optional] mini-batch size default=500
        :return:
        """
        y_obj = []
        y_rel_box = []
        for batch in self.trainer.batch_iterator(X, batchsize=batchsize, shuffle=False):
            if batch_xform_fn is not None:
                batch = batch_xform_fn(batch)
            y_obj_prob_batch, y_rel_box_batch = self._predict_prob_fn(batch[0])
            y_obj_batch = np.argmax(y_obj_prob_batch, axis=1)
            y_obj_batch = y_obj_batch.reshape((-1, self.n_anchor_boxes))
            y_obj.append(y_obj_batch)
            y_rel_box.append(y_rel_box_batch)
        y_obj = np.concatenate(y_obj, axis=0)
        y_rel_box = np.concatenate(y_rel_box, axis=0)
        return y_obj, y_rel_box

