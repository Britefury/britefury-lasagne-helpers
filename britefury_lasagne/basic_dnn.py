import six
import numpy as np
from functools import partial
import theano
import theano.tensor as T
import lasagne
from batchup import data_source
from . import trainer, dnn_objective


def _is_sequence_of_layers(xs):
    if isinstance(xs, (tuple, list)):
        for x in xs:
            if not isinstance(x, lasagne.layers.Layer):
                return False
        return True
    return False

def _get_network_params(layer, updates=None):
    layer_params = lasagne.layers.get_all_params(layer)

    if updates is not None:
        if isinstance(updates, dict):
            params = list(updates.keys())
        elif isinstance(updates, (list, tuple)):
            params = [upd[0] for upd in updates]
        else:
            raise TypeError('updates should be a dict mapping parameter to update expression '
                            'or a sequence of tuples of (parameter, update_expression) pairs')

        for p in params:
            if p not in layer_params:
                layer_params.append(p)

    return layer_params


class BasicDNN (object):
    def __init__(self, input_vars, target_and_mask_vars, final_layers, objectives,
                 score_objective=None,
                 trainable_params=None, updates_fn=None, params_source=None):
        """
        Constructor - construct a `SampleDNN` instance given variables for
        input, target and a final layer (a Lasagne layer)
        :param input_vars: input variables, a list of Theano variables
        :param target_and_mask_vars: target and mask variables, a list of Theano variables
        :param final_layers: a list of Lasagne layers that when followed backward will
                result in all layers being visited
        :param objectives: a list of objectives to optimise
        :param score_objective: the objective that is used to measure the performance of the network
                for the purpose of early stopping
        :param trainable_params: [optional] a list of parameters to train. If `None`, then the
                parameters from all the layers in the network will be used
        :param updates_fn: [optional] a function of the form `fn(cost, params) -> updates` that
            generates update expressions given the cost and the parameters to update using
            an optimisation technique e.g. Nesterov Momentum:
            `lambda cost, params: lasagne.updates.nesterov_momentum(cost, params,
                learning_rate=0.002, momentum=0.9)`
        :param params_source: [optional] source from which to obtain network parameters; either
            a str/unicode that contains the path of a NumPy array file from which to load the parameters,
            or a `BasicDNN` or Lasagne layer from which to copy the parameters
        """
        self.input_vars = input_vars
        self.target_and_mask_vars = target_and_mask_vars
        self.final_layers = final_layers
        self.objectives = objectives

        if score_objective is None:
            score_objective = objectives[0]
        else:
            if score_objective not in objectives:
                raise ValueError('score_objective not in objectives')

        self.score_objective = score_objective

        if params_source is not None:
            if isinstance(params_source, six.string_types):
                self.load_params(params_source)
            elif isinstance(params_source, BasicDNN):
                self.set_param_values(params_source.get_param_values())
            elif isinstance(params_source, lasagne.layers.Layer) or _is_sequence_of_layers(params_source):
                params_in = _get_network_params(params_source)
                values = [p.get_value() for p in params_in]
                self.set_param_values(values)
            else:
                raise TypeError('params_source must be a string containing a path, a `BasicDNN` instance, '
                                'a Lasagne layer or a sequence of Lasagne layers, not a {}'.format(type(params_source)))

        self.objective_results = [obj.build() for obj in self.objectives]
        train_cost = sum([obj_res.train_cost for obj_res in self.objective_results])
        train_results = []
        self.train_results_indices = []
        eval_results = []
        self.eval_results_indices = []
        self.score_objective_index = self.objectives.index(score_objective)
        predictions = []
        for obj_res in self.objective_results:
            self.train_results_indices.append(len(train_results))
            train_results.extend(obj_res.train_results)
            self.eval_results_indices.append(len(eval_results))
            eval_results.extend(obj_res.eval_results)
            predictions.append(obj_res.prediction)
        self.train_results_indices.append(len(train_results))
        self.eval_results_indices.append(len(eval_results))

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        if trainable_params is None:
            trainable_params = lasagne.layers.get_all_params(final_layers, trainable=True)
        if updates_fn is None:
            self._updates = lasagne.updates.adam(
                    train_cost, trainable_params, learning_rate=0.001)
        else:
            self._updates = updates_fn(train_cost, trainable_params)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self._train_fn = theano.function(input_vars + target_and_mask_vars, train_results, updates=self._updates)

        # Compile a function computing the validation loss and error:
        self._val_fn = theano.function(input_vars + target_and_mask_vars, eval_results)

        # Compile a function computing the predicted probability
        self._predict_fn = theano.function(input_vars, predictions)

        # Construct a training function
        self.train = partial(trainer.train,
                             train_batch_func=self._train_fn, train_log_msg=self._train_log,
                             train_epoch_results_check_func=self._check_train_epoch_results,
                             eval_batch_func=self._val_fn, eval_log_msg=self._eval_log,
                             val_improved_func=self._score_improved,
                             epoch_log_msg=self._epoch_log, layer_to_restore=final_layers)


    def load_params(self, params_path, include_updates=False):
        """
        Load parameters from an NPZ file found at the specified path
        :param params_path: path of file from which to load parameters
        """
        updates = self._updates if include_updates else None
        with np.load(params_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        params = _get_network_params(self.final_layers, updates=updates)
        for p, v in zip(params, param_values):
            p.set_value(v)

    def save_params(self, params_path, include_updates=False):
        """
        Save parameters to an NPZ file found at the specified path
        :param params_path: path of file to save parameters to
        """
        updates = self._updates if include_updates else None
        params = _get_network_params(self.final_layers, updates=updates)
        param_values = [p.get_value() for p in params]
        np.savez(params_path, *param_values)


    def get_param_values(self, include_updates=False):
        """
        Get the values of all parameters in the network
        :return: a list of NumPy arrays
        """
        params = self.get_params(include_updates)
        return [p.get_value() for p in params]

    def set_param_values(self, values, include_updates=False):
        """
        Set the values of all parameters in the network.

        Walks the network and extracts a list containing all parameters, then updates
        each parameter with the value in the corresponding position in `values`.

        :param values: a list of NumPy arrays that contain the values for the parameters
        """
        params = self.get_params(include_updates)
        for p, v in zip(params, values):
            p.set_value(v)

    def get_params(self, include_updates=False):
        """
        Get all parameters in the network
        :param include_updates: if True, include parameters used for updates
        :return: a list of Theano shared variables
        """
        updates = self._updates if include_updates else None
        return _get_network_params(self.final_layers, updates=updates)

    def get_param_names(self, include_updates=False):
        """
        Get the names of all parameters in the network
        :return: a list of strings
        """
        return [p.name for p in self.get_params(include_updates=include_updates)]


    def _check_train_epoch_results(self, epoch, train_epoch_results):
        if np.isnan(train_epoch_results).any():
            return 'Training loss of NaN'
        else:
            return None


    def _train_log(self, train_results):
        train_items = []
        for obj_res, i, j in zip(self.objective_results, self.train_results_indices[:-1],
                                 self.train_results_indices[1:]):
            train_items.append(obj_res.train_results_str_fn(train_results[i:j]))
        return ', '.join(train_items)

    def _eval_log(self, eval_results):
        eval_items = []
        for obj_res, i, j in zip(self.objective_results, self.eval_results_indices[:-1],
                                 self.eval_results_indices[1:]):
            eval_items.append(obj_res.eval_results_str_fn(eval_results[i:j]))
        return ', '.join(eval_items)

    def _score_improved(self, new_results, best_so_far):
        i, j = self.eval_results_indices[self.score_objective_index:self.score_objective_index+2]
        new_obj_res = new_results[i:j]
        best_obj_res = best_so_far[i:j]
        return self.score_objective.score_improved(new_obj_res, best_obj_res)

    def _epoch_log(self, epoch_number, delta_time, train_str, val_str, test_str):
        """
        Epoch logging callback; used to build the training function`
        """
        epoch_n = (epoch_number + 1) if epoch_number is not None else '<final>'
        items = []
        items.append('Epoch {} took {:.2f}s:'.format(epoch_n, delta_time))
        items.append('  TRAIN ')

        if train_str is not None:
            items.append(train_str)

        if val_str is not None:
            items.append('  VAL ')
            items.append(val_str)

        if test_str is not None:
            items.append('  TEST ')
            items.append(test_str)

        return ''.join(items)


    def predict(self, X, batchsize=500):
        """
        Evaluate the network, returning its predictions

        :param X: input data as a data source
        :param batchsize: the mini-batch size
        :return: a list of predicted outputs, where each entry corresponds to a training objective
        e.g. a simple classifier will return the list `[pred_prob]` where `pred_prob` is the predicted class
        probabilities
        """
        return data_source.coerce_data_source(X).batch_map_concat(self._predict_fn, batch_size=batchsize)


class BasicClassifierDNN (BasicDNN):
    """
    A simple classifier DNN; its purpose is to provide the `temperature` property for
    changing the temperature of the softmax nonlinearity in order to 'soften' predicted
    probabilities.
    """
    def __init__(self, input_vars, target_and_mask_vars, final_layers, classifier_objective,
                 score_objective=None,
                 trainable_params=None, updates_fn=None, params_source=None):
        if not isinstance(classifier_objective, dnn_objective.ClassifierObjective):
            raise TypeError('classifier_objective must be an instance of dnn_objective.ClassifierObjective')
        super(BasicClassifierDNN, self).__init__(input_vars, target_and_mask_vars, final_layers,
                                                 [classifier_objective], score_objective=score_objective,
                                                 trainable_params=trainable_params, updates_fn=updates_fn,
                                                 params_source=params_source)
        self._classifier_objective = classifier_objective

    @property
    def temperature(self):
        return self._classifier_objective.temperature

    @temperature.setter
    def temperature(self, t):
        self._classifier_objective.temperature = t

    def predict(self, X, batchsize=500, temperature=None):
        """
        Evaluate the network, returning its predictions

        :param X: input data as a data source
        :param batchsize: the mini-batch size
        :return: a list of predicted outputs, where each entry corresponds to a training objective
        e.g. a simple classifier will return the list `[pred_prob]` where `pred_prob` is the predicted class
        probabilities
        """
        old_temperature = None
        if temperature is not None:
            old_temperature = self.temperature
            self.temperature = temperature
        res = super(BasicClassifierDNN, self).predict(X, batchsize=batchsize)
        if temperature is not None:
            self.temperature = old_temperature

        return res


def _get_input_layers(final_layer):
    layers = lasagne.layers.get_all_layers(final_layer)
    return [layer for layer in layers if isinstance(layer, lasagne.layers.InputLayer)]

def _get_input_vars(final_layer):
    input_layers = _get_input_layers(final_layer)
    return [layer.input_var for layer in input_layers]

def simple_classifier(network_build_fn, n_input_spatial_dims=0, n_target_spatial_dims=0,
                      target_channel_index=None, score=dnn_objective.ClassifierObjective.SCORE_ERROR, mask=False,
                      includes_softmax=False, params_source=None, *args, **kwargs):
    """
    Construct an image classifier, given a network building function
    and an optional path from which to load parameters.
    :param network_build_fn: network builder function of the form `fn(input_vars) -> lasagne_layer`
        that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the input;
        0 for input sample with matrix variable type (sample, channel)
        1 for 1-dimensional input e.g. time series, with tensor3 variable type (sample, channel, time),
        2 for 2-dimensional input e.g. image, with tensor4 variable type (sample, channel, height, width),
        3 for 3-dimensional input e.g. volume, with tensor5 variable type (sample, channel, depth, height, width),
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with ivector variable type
        1 for 1-dimensional prediction e.g. time series, with imatrix variable type (sample, time),
        2 for 2-dimensional prediction e.g. image, with itensor3 variable type (sample, height, width),
        3 for 3-dimensional prediction e.g. volumn, with itensor4 variable type (sample, depth, height, width),
    :param target_channel_index: if None, targets are assumed not to have a channel dimension. If an integer,
        then this channel will be used for the target, e.g.
        for a target with 0 spatial dimensions, if `target_channel_index` is `None` then the targets
        should have shape `(sample,)`, while if there are 5 channels and the target uses channel 2,
        the target should have shape `(sample, 5)` and we will access the target indices in channel, e.g. `y[:,2]`.
        Note that the additional channel dimension adds an additional dimension to target and mask variables, e.g.
        0, 1, 2 and 3 dimensional targets and masks use imatrix, itensor3, itensor4 and itensor5 variable types.
    :param score: the scoring metric used to evaluate classifier performance (see `dnn_objective.ClassifierObjective`)
    :param mask: (default=False) if True, samples will be masked, in which case sample weights/masks should
        be passed during training
    :param includes_softmax: `True` indicates that the final network layer includes the softmax non-linearity,
        `False` indicates that it does not, in which case a non-linearity layer will be added
    :param params_source: [optional] source from which to obtain network parameters; either
        a str/unicode that contains the path of a NumPy array file from which to load the parameters,
        or a `BasicDNN` or Lasagne layer from which to copy the parameters
    :return: a classifier instance
    """
    if n_input_spatial_dims == 0:
        input_vars = [T.matrix('x')]
    elif n_input_spatial_dims == 1:
        input_vars = [T.tensor3('x')]
    elif n_input_spatial_dims == 2:
        input_vars = [T.tensor4('x')]
    elif n_input_spatial_dims == 3:
        tensor5 = T.TensorType(theano.config.floatX, (False,)*5, 'tensor5')
        input_vars = [tensor5('x')]
    else:
        raise ValueError('Valid values for n_input_spatial_dims are in the range 0-3, not {}'.format(
            n_target_spatial_dims))
    return classifier(input_vars, network_build_fn, n_target_spatial_dims=n_target_spatial_dims,
                      target_channel_index=target_channel_index, score=score, mask=mask,
                      includes_softmax=includes_softmax, params_source=params_source, *args, **kwargs)


def classifier(input_vars, network_build_fn, n_target_spatial_dims=0, target_channel_index=None,
               score=dnn_objective.ClassifierObjective.SCORE_ERROR, mask=False, includes_softmax=False,
               params_source=None, *args, **kwargs):
    """
    Construct a classifier, given input variables and a network building function
    and an optional path from which to load parameters.
    :param input_vars: a list of input variables. If `None`, the network will be searched for `InputLayer` instances
        and their input variables will be used.
    :param network_build_fn: network builder function of the form `fn(input_vars) -> lasagne_layer`
        that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with ivector variable type
        1 for 1-dimensional prediction e.g. time series, with imatrix variable type (sample, time),
        2 for 2-dimensional prediction e.g. image, with itensor3 variable type (sample, height, width),
        3 for 3-dimensional prediction e.g. volume, with itensor4 variable type (sample, depth, height, width),
    :param target_channel_index: if None, targets are assumed not to have a channel dimension. If an integer,
        then this channel will be used for the target, e.g.
        for a target with 0 spatial dimensions, if `target_channel_index` is `None` then the targets
        should have shape `(sample,)`, while if there are 5 channels and the target uses channel 2,
        the target should have shape `(sample, 5)` and we will access the target indices in channel, e.g. `y[:,2]`.
        Note that the additional channel dimension adds an additional dimension to target and mask variables, e.g.
        0, 1, 2 and 3 dimensional targets and masks use imatrix, itensor3, itensor4 and itensor5 variable types.
    :param score: the scoring metric used to evaluate classifier performance (see `dnn_objective.ClassifierObjective`)
    :param mask: (default=False) if True, samples will be masked, in which case sample weights/masks should
        be passed during training
    :param includes_softmax: `True` indicates that the final network layer includes the softmax non-linearity,
        `False` indicates that it does not, in which case a non-linearity layer will be added
    :param params_source: [optional] source from which to obtain network parameters; either
        a str/unicode that contains the path of a NumPy array file from which to load the parameters,
        or a `BasicDNN` or Lasagne layer from which to copy the parameters
    :return: a classifier instance
    """
    # Prepare Theano variables for inputs and targets
    n_target_tims = n_target_spatial_dims + (0 if target_channel_index is None else 1)
    if n_target_tims == 0:
        target_var = T.ivector('y')
    elif n_target_tims == 1:
        target_var = T.imatrix('y')
    elif n_target_tims == 2:
        target_var = T.itensor3('y')
    elif n_target_tims == 3:
        target_var = T.itensor4('y')
    else:
        raise ValueError('Valid values for n_target_spatial_dims are in the range 0-3, not {}'.format(
            n_target_spatial_dims))

    if mask:
        if n_target_tims == 0:
            mask_var = T.vector('m')
        elif n_target_tims == 1:
            mask_var = T.matrix('m')
        elif n_target_tims == 2:
            mask_var = T.tensor3('m')
        elif n_target_tims == 3:
            mask_var = T.tensor4('m')
        else:
            raise ValueError('Valid values for n_target_spatial_dims are in the range 0-3, not {}'.format(
                n_target_spatial_dims))
        mask_vars = [mask_var]
    else:
        mask_var = None
        mask_vars = []


    # Build the network
    network = network_build_fn(input_vars=input_vars)
    if input_vars is None:
        input_vars = _get_input_vars(network)

    objective = dnn_objective.ClassifierObjective('y', network, target_var, mask_expr=mask_var,
                                                  n_target_spatial_dims=n_target_spatial_dims,
                                                  target_channel_index=target_channel_index, score=score,
                                                  includes_softmax=includes_softmax)

    return BasicClassifierDNN(input_vars, [target_var] + mask_vars, network, objective,
                              params_source=params_source, *args, **kwargs)

def simple_regressor(network_build_fn, n_input_spatial_dims=0, n_target_spatial_dims=0, mask=False,
                     params_source=None, *args, **kwargs):
    """
    Construct a vector regressor, given a network building function
    and an optional path from which to load parameters.
    :param network_build_fn: network builder function of the form `fn(input_vars) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the input;
        0 for input sample with matrix variable type (sample, channel)
        1 for 1-dimensional input e.g. time series, with tensor3 variable type (sample, channel, time),
        2 for 2-dimensional input e.g. image, with tensor4 variable type (sample, channel, height, width),
        3 for 3-dimensional input e.g. volume, with tensor5 variable type (sample, channel, depth, height, width),
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with matrix variable type (sample, channel)
        1 for 1-dimensional prediction e.g. time series, with tensor3 variable type (sample, channel, time),
        2 for 2-dimensional prediction e.g. image, with tensor4 variable type (sample, channel, height, width),
        3 for 3-dimensional prediction e.g. volume, with tensor5 variable type (sample, channel, depth, height, width),
    :param mask: (default=False) if True, samples will be masked, in which case sample weights/masks should
    be passed during training
    :param params_source: [optional] source from which to obtain network parameters; either
        a str/unicode that contains the path of a NumPy array file from which to load the parameters,
        or a `BasicDNN` or Lasagne layer from which to copy the parameters
    :return: a classifier instance
    """
    if n_input_spatial_dims == 0:
        input_vars = [T.matrix('x')]
    elif n_input_spatial_dims == 1:
        input_vars = [T.tensor3('x')]
    elif n_input_spatial_dims == 2:
        input_vars = [T.tensor4('x')]
    elif n_input_spatial_dims == 3:
        tensor5 = T.TensorType(theano.config.floatX, (False,)*5, 'tensor5')
        input_vars = [tensor5('x')]
    else:
        raise ValueError('Valid values for n_input_spatial_dims are in the range 0-3, not {}'.format(
            n_target_spatial_dims))
    return regressor(input_vars, network_build_fn, n_target_spatial_dims=n_target_spatial_dims,
                     mask=mask, params_source=params_source, *args, **kwargs)


def regressor(input_vars, network_build_fn, n_target_spatial_dims=0, mask=False,
              params_source=None, *args, **kwargs):
    """
    Construct a regressor, given a network building function
    and an optional path from which to load parameters.
    :param input_vars: a list of input variables. If `None`, the network will be searched for `InputLayer` instances
        and their input variables will be used.
    :param network_build_fn: network builder function of the form `fn(input_vars) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with matrix variable type (sample, channel)
        1 for 1-dimensional prediction e.g. time series, with tensor3 variable type (sample, channel, time),
        2 for 2-dimensional prediction e.g. image, with tensor4 variable type (sample, channel, height, width),
        3 for 3-dimensional prediction e.g. volume, with tensor5 variable type (sample, channel, depth, height, width),
    :param mask: (default=False) if True, samples will be masked, in which case sample weights/masks should
    be passed during training
    :param params_source: [optional] source from which to obtain network parameters; either
        a str/unicode that contains the path of a NumPy array file from which to load the parameters,
        or a `BasicDNN` or Lasagne layer from which to copy the parameters
    :return: a classifier instance
    """
    # Prepare Theano variables for inputs and targets
    tensor5 = T.TensorType(theano.config.floatX, (False,)*5, 'tensor5')
    if n_target_spatial_dims == 0:
        target_var = T.matrix('y')
    elif n_target_spatial_dims == 1:
        target_var = T.tensor3('y')
    elif n_target_spatial_dims == 2:
        target_var = T.tensor4('y')
    elif n_target_spatial_dims == 3:
        target_var = tensor5('y')
    else:
        raise ValueError('Valid values for n_target_spatial_dims are in the range 0-3, not {}'.format(
            n_target_spatial_dims))

    if mask:
        if n_target_spatial_dims == 0:
            mask_var = T.matrix('m')
        elif n_target_spatial_dims == 1:
            mask_var = T.tensor3('m')
        elif n_target_spatial_dims == 2:
            mask_var = T.tensor4('m')
        elif n_target_spatial_dims == 3:
            mask_var = tensor5('m')
        else:
            raise ValueError('Valid values for n_target_spatial_dims are in the range 0-3, not {}'.format(
                n_target_spatial_dims))
        mask_vars = [mask_var]
    else:
        mask_var = None
        mask_vars = []

    # Build the network
    network = network_build_fn(input_vars=input_vars)
    if input_vars is None:
        input_vars = _get_input_vars(network)

    objective = dnn_objective.RegressorObjective('y', network, target_var, mask_expr=mask_var,
                                                 n_target_spatial_dims=n_target_spatial_dims)

    return BasicDNN(input_vars, [target_var] + mask_vars, network, [objective],
                    params_source=params_source, *args, **kwargs)
