import numpy as np
import theano
import theano.tensor as T
import lasagne
from . import trainer, dnn_objective


class BasicDNN (object):
    def __init__(self, input_vars, target_vars, final_layers, objectives, updates_fn=None,
                 params_path=None):
        """
        Constructor - construct a `SampleDNN` instance given variables for
        input, target and a final layer (a Lasagne layer)
        :param input_vars: input variables, a list of Theano variables
        :param final_layers: a list of Lasagne layers that when followed backward will cover
                result in all layers being visited
        :param objectives: a list of objectives to optimise
        :param updates_fn: [optional] a function of the form `fn(cost, params) -> updates` that
            generates update expressions given the cost and the parameters to update using
            an optimisation technique e.g. Nesterov Momentum:
            `lambda cost, params: lasagne.updates.nesterov_momentum(cost, params,
                learning_rate=0.002, momentum=0.9)`
        :param params_path: [optional] path from which to load network parameters
        """
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.final_layers = final_layers
        self.objectives = objectives

        if params_path is not None:
            trainer.load_model(params_path, final_layers)

        self.objective_results = [obj.build() for obj in self.objectives]
        train_cost = sum([obj_res.train_cost for obj_res in self.objective_results])
        train_results = []
        self.train_results_indices = []
        eval_results = []
        self.eval_results_indices = []
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
        params = lasagne.layers.get_all_params(final_layers, trainable=True)
        if updates_fn is None:
            updates = lasagne.updates.nesterov_momentum(
                    train_cost, params, learning_rate=0.01, momentum=0.9)
        else:
            updates = updates_fn(train_cost, params)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self._train_fn = theano.function(input_vars + target_vars, train_results, updates=updates)

        # Compile a function computing the validation loss and error:
        self._val_fn = theano.function(input_vars + target_vars, eval_results)

        # Compile a function computing the predicted probability
        self._predict_fn = theano.function(input_vars, predictions)


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
        self.trainer.retain_best_scoring_state_of_network(final_layers)


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
        items.append('Epoch {}/{} took {:.2f}s:'.format(epoch_number + 1, self.trainer.num_epochs, delta_time))
        items.append('  TRAIN ')

        train_items = []
        for obj_res, i, j in zip(self.objective_results, self.train_results_indices[:-1],
                                 self.train_results_indices[1:]):
            train_items.append(obj_res.train_results_str_fn(train_results[i:j]))
        items.append(', '.join(train_items))

        if val_results is not None:
            val_items = []
            items.append('  VAL ')
            for obj_res, i, j in zip(self.objective_results, self.eval_results_indices[:-1],
                                     self.eval_results_indices[1:]):
                val_items.append(obj_res.eval_results_str_fn(val_results[i:j]))
            items.append(', '.join(val_items))

        if test_results is not None:
            test_items = []
            items.append('  TEST ')
            for obj_res, i, j in zip(self.objective_results, self.eval_results_indices[:-1],
                                     self.eval_results_indices[1:]):
                test_items.append(obj_res.eval_results_str_fn(test_results[i:j]))
            items.append(', '.join(test_items))

        return ''.join(items)


    def predict(self, X, batchsize=500, batch_xform_fn=None):
        y = []
        for batch in self.trainer.batch_iterator(X, batchsize=batchsize, shuffle=False):
            if batch_xform_fn is not None:
                batch = batch_xform_fn(batch)
            y_batch = self._predict_fn(batch[0])
            y.append(y_batch)
        return [np.concatenate(chn, axis=0) for chn in zip(*y)]


def vector_classifier(network_build_fn, n_target_spatial_dims=0, params_path=None, *args, **kwargs):
    """
    Construct a vector classifier, given a network building function
    and an optional path from which to load parameters.
    :param network_build_fn: network builder function of the form `fn(input_var, **kwargs) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of dimensions in the target;
        0 for predict per sample with ivector variable type
        1 for 1-dimensional prediction e.g. time series, with itensor3 variable type (sample, channel (1), time),
        2 for 2-dimensional prediction e.g. image, with itensor4 variable type (sample, channel (1), height, width),
    :param params_path: [optional] path from which to load network parameters
    :return: a classifier instance
    """
    input_vars = [T.matrix('input')]
    return classifier(input_vars, network_build_fn, n_target_spatial_dims=n_target_spatial_dims,
                      params_path=params_path)


def image_classifier(network_build_fn, n_target_spatial_dims=0, params_path=None, *args, **kwargs):
    """
    Construct an image classifier, given a network building function
    and an optional path from which to load parameters.
    :param network_build_fn: network builder function of the form `fn(input_var, **kwargs) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with ivector variable type
        1 for 1-dimensional prediction e.g. time series, with imatrix variable type (sample, channel (1), time),
        2 for 2-dimensional prediction e.g. image, with itensor3 variable type (sample, channel (1), height, width),
    :param params_path: [optional] path from which to load network parameters
    :return: a classifier instance
    """
    input_vars = [T.tensor4('input')]
    return classifier(input_vars, network_build_fn, n_target_spatial_dims=n_target_spatial_dims,
                      params_path=params_path)


def classifier(input_vars, network_build_fn, n_target_spatial_dims=0, params_path=None, *args, **kwargs):
    """
    Construct a classifier, given input variables and a network building function
    and an optional path from which to load parameters.
    :param input_vars: a list of input variables
    :param network_build_fn: network builder function of the form `fn(input_var, **kwargs) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with ivector variable type
        1 for 1-dimensional prediction e.g. time series, with imatrix variable type (sample, channel (1), time),
        2 for 2-dimensional prediction e.g. image, with itensor3 variable type (sample, channel (1), height, width),
    :param params_path: [optional] path from which to load network parameters
    :return: a classifier instance
    """
    # Prepare Theano variables for inputs and targets
    if n_target_spatial_dims == 0:
        target_var = T.ivector('y')
    elif n_target_spatial_dims == 1:
        target_var = T.itensor3('y')
    elif n_target_spatial_dims == 2:
        target_var = T.itensor4('y')
    elif n_target_spatial_dims == 3:
        itensor5 = T.TensorType('int32', (False,)*5, 'itensor5')
        target_var = itensor5('y')
    else:
        raise ValueError('Valid values for n_target_dim are 0, 1, or 2, not {}'.format(n_target_spatial_dims))

    # Build the network
    print("Building model and compiling functions...")
    network = network_build_fn(input_vars=input_vars)

    objective = dnn_objective.ClassifierObjective('y', network, target_var)

    return BasicDNN(input_vars, [target_var], network, [objective], params_path=params_path, *args, **kwargs)


def vector_regressor(network_build_fn, n_target_spatial_dims=0, params_path=None, *args, **kwargs):
    """
    Construct a vector regressor, given a network building function
    and an optional path from which to load parameters.
    :param network_build_fn: network builder function of the form `fn(input_var, **kwargs) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with matrix variable type (sample, channel)
        1 for 1-dimensional prediction e.g. time series, with itensor3 variable type (sample, channel, time),
        2 for 2-dimensional prediction e.g. image, with itensor4 variable type (sample, channel, height, width),
    :param params_path: [optional] path from which to load network parameters
    :return: a classifier instance
    """
    input_vars = [T.matrix('input')]
    return regressor(input_vars, network_build_fn, n_target_spatial_dims=n_target_spatial_dims,
                     params_path=params_path)


def image_regressor(network_build_fn, n_target_spatial_dims=0, params_path=None, *args, **kwargs):
    """
    Construct an image regressor, given a network building function
    and an optional path from which to load parameters.
    :param network_build_fn: network builder function of the form `fn(input_var, **kwargs) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with matrix variable type (sample, channel)
        1 for 1-dimensional prediction e.g. time series, with itensor3 variable type (sample, channel, time),
        2 for 2-dimensional prediction e.g. image, with itensor4 variable type (sample, channel, height, width),
    :param params_path: [optional] path from which to load network parameters
    :return: a classifier instance
    """
    input_vars = [T.tensor4('input')]
    return regressor(input_vars, network_build_fn, n_target_spatial_dims=n_target_spatial_dims,
                     params_path=params_path)


def regressor(input_vars, network_build_fn, n_target_spatial_dims=0, params_path=None, *args, **kwargs):
    """
    Construct a regressor, given a network building function
    and an optional path from which to load parameters.
    :param input_vars: a list of input variables
    :param network_build_fn: network builder function of the form `fn(input_var, **kwargs) -> lasagne_layer`
    that constructs a network in the form of a Lasagne layer, given an input variable (a Theano variable)
    :param n_target_spatial_dims: the number of spatial dimensions for the target;
        0 for predict per sample with matrix variable type (sample, channel)
        1 for 1-dimensional prediction e.g. time series, with itensor3 variable type (sample, channel, time),
        2 for 2-dimensional prediction e.g. image, with itensor4 variable type (sample, channel, height, width),
    :param params_path: [optional] path from which to load network parameters
    :return: a classifier instance
    """
    # Prepare Theano variables for inputs and targets
    if n_target_spatial_dims == 0:
        target_var = T.matrix('y')
    elif n_target_spatial_dims == 1:
        target_var = T.tensor3('y')
    elif n_target_spatial_dims == 2:
        target_var = T.tensor4('y')
    elif n_target_spatial_dims == 3:
        tensor5 = T.TensorType(theano.config.floatX, (False,)*5, 'itensor5')
        target_var = tensor5('y')
    else:
        raise ValueError('Valid values for n_target_dim are 0, 1, or 2, not {}'.format(n_target_spatial_dims))


    # Build the network
    print("Building model and compiling functions...")
    network = network_build_fn(input_vars=input_vars)

    objective = dnn_objective.RegressorObjective('y', network, target_var)

    return BasicDNN(input_vars, [target_var], network, [objective], params_path=params_path, *args, **kwargs)
