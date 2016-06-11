import numpy as np
import theano
import theano.tensor as T

import lasagne


class TemperatureSoftmax (object):
    """
    A softmax function with a temperature setting; increasing it smooths the resulting probabilities.
    """
    def __init__(self, temperature=1.0):
        self._temperature = theano.shared(lasagne.utils.floatX(temperature), 'temperature')

    @property
    def temperature(self):
        return 1.0 / self._temperature.get_value()

    @temperature.setter
    def temperature(self, value):
        self._temperature.set_value(lasagne.utils.floatX(1.0 / value))

    def __call__(self, x):
        return lasagne.nonlinearities.softmax(x * self._temperature)


def _flatten_spatial_lasagne_layer(layer):
    out_shape = layer.output_shape
    ndims = len(out_shape)
    spatial = tuple(range(2, ndims))
    n_channels = out_shape[1]
    spatial_shape = out_shape[2:]
    if ndims > 2:
        # (Sample,Channel,Spatial...) -> (Channel,Sample,Spatial...)
        layer = lasagne.layers.DimshuffleLayer(layer, (1, 0) + spatial)
        # (Channel,Sample,spatial...) -> (Channel,Sample:spatial...)
        layer = lasagne.layers.FlattenLayer(layer, outdim=2)
        # (Channel,Sample:spatial...) -> (Sample:spatial...,Channel)
        layer = lasagne.layers.DimshuffleLayer(layer, (1, 0))
    return layer, spatial_shape, n_channels

def _flatten_spatial_theano(x, n_channels):
    ndim = x.ndim
    if n_channels is None:
        if ndim > 1:
            # (Sample,Spatial...) -> (Sample:Spatial...)
            x = x.reshape((-1,))
    else:
        if ndim > 2:
            spatial = tuple(range(2, ndim))
            # (Sample,Channel,Spatial...) -> (Sample,Spatial...,Channel)
            x = x.dimshuffle((0,) + spatial + (1,))
            # (Sample,Spatial...,Channel) -> (Sample:Spatial...,Channel)
            x = x.reshape((-1, n_channels))
    return x

def _unflatten_spatial_theano(x, spatial_shape, n_channels):
    n_spatial = len(spatial_shape)
    if n_spatial > 0:
        if n_channels is None:
            # (Sample:spatial...) -> (Sample,Spatial...)
            x = x.reshape((-1,) + spatial_shape)
        else:
            # (Sample:spatial...,Channel) -> (Sample,Spatial...,Channel)
            x = x.reshape((-1,) + spatial_shape + (n_channels,))
            # (Sample,Spatial...,Channel) -> (Sample,Channel,Spatial...)
            x = x.dimshuffle((0, n_spatial+1,) + tuple(range(1,n_spatial+1)))
    return x

def _flatten_spatial(x, n_channels):
    ndim = len(x.shape)
    if n_channels is None:
        if ndim > 1:
            # (Sample,Spatial...) -> (Sample:Spatial...)
            x = x.reshape((-1,))
    else:
        if ndim > 2:
            # (Sample,Channel,Spatial...) -> (Sample,Spatial...,Channel)
            x = np.rollaxis(x, 1, ndim)
            # (Sample,Spatial...,Channel) -> (Sample:Spatial...,Channel)
            x = x.reshape((-1, n_channels))
    return x

def _unflatten_spatial(x, spatial_shape, n_channels):
    n_spatial = len(spatial_shape)
    if n_spatial > 0:
        if n_channels is None:
            # (Sample:spatial...) -> (Sample,Spatial...)
            x = x.reshape((-1,) + spatial_shape)
        else:
            # (Sample:spatial...,Channel) -> (Sample,Spatial...,Channel)
            x = x.reshape((-1,) + spatial_shape + (n_channels,))
            # (Sample,Spatial...,Channel) -> (Sample,Channel,Spatial...)
            x = np.rollaxis(x, 3, 1)
    return x


class ObjectiveOutput (object):
    def __init__(self, train_cost, train_results, train_results_str_fn,
                 eval_results, eval_results_str_fn, prediction):
        self.train_cost = train_cost
        self.train_results = train_results
        self.train_results_str_fn = train_results_str_fn
        self.eval_results = eval_results
        self.eval_results_str_fn = eval_results_str_fn
        self.prediction = prediction


class AbstractObjective (object):
    transform_input_target = None
    inv_transform_prediction = None

    def __init__(self, name, cost_weight=1.0):
        self.name = name
        self.cost_weight = lasagne.utils.floatX(cost_weight)

    def build(self):
        raise NotImplementedError('Abstract for {}'.format(type(self)))



class ClassifierObjective (AbstractObjective):
    def __init__(self, name, objective_layer, target_expr, cost_weight=1.0):
        super(ClassifierObjective, self).__init__(name, cost_weight)
        self.objective_layer = objective_layer
        self.target_expr = target_expr


    def build(self):
        flat_target = _flatten_spatial_theano(self.target_expr, None)
        if flat_target.ndim != 1:
            raise ValueError('target must have 1 dimensions, not {}'.format(flat_target.ndim))

        # Flatten the objective layer (if the objective layer generates an
        # output with 2 dimensions then this is a no-op)
        obj_flat_layer, spatial_shape, n_out_channels = \
            _flatten_spatial_lasagne_layer(self.objective_layer)

        # Predicted probability layer
        softmax = TemperatureSoftmax()
        prob_layer = lasagne.layers.NonlinearityLayer(obj_flat_layer, softmax)

        # Get an expression representing the predicted probability
        train_pred_prob = lasagne.layers.get_output(prob_layer)

        # Create a per-sample loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        train_loss_per_sample = lasagne.objectives.categorical_crossentropy(train_pred_prob, flat_target)

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_pred_prob = lasagne.layers.get_output(prob_layer, deterministic=True)
        # Create evaluation loss expression
        eval_loss_per_sample = lasagne.objectives.categorical_crossentropy(eval_pred_prob, flat_target)
        # Create an expression for error count
        eval_err_count = T.sum(T.neq(T.argmax(eval_pred_prob, axis=1), flat_target),
                                     dtype=theano.config.floatX)

        # Unflatten prediction
        pred_prob = _unflatten_spatial_theano(eval_pred_prob, spatial_shape, n_out_channels)

        def train_results_str_fn(train_res):
            return '{} loss={:.6f}'.format(self.name, train_res[0])

        def eval_results_str_fn(eval_res):
            return '{} loss={:.6f} err={:.2%}'.format(self.name, eval_res[0], eval_res[1])

        return ObjectiveOutput(train_cost=train_loss_per_sample.mean() * self.cost_weight,
                               train_results=[train_loss_per_sample.sum()],
                               train_results_str_fn=train_results_str_fn,
                               eval_results=[eval_loss_per_sample.sum(), eval_err_count],
                               eval_results_str_fn=eval_results_str_fn,
                               prediction=pred_prob)


class RegressorObjective (AbstractObjective):
    def __init__(self, name, objective_layer, target_expr, cost_weight=1.0):
        super(RegressorObjective, self).__init__(name, cost_weight)
        self.objective_layer = objective_layer
        self.target_expr = target_expr


    def build(self):
        softmax = TemperatureSoftmax()

        # Get an expression representing the predicted probability
        train_pred = lasagne.layers.get_output(self.objective_layer)

        # Create a per-sample loss expression for training, i.e., a scalar objective we want
        # to minimize; squared error
        train_loss_per_sample = lasagne.objectives.squared_error(train_pred, self.target_expr)

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_pred = lasagne.layers.get_output(self.objective_layer, deterministic=True)
        # Create evaluation loss expression
        eval_loss_per_sample = lasagne.objectives.squared_error(eval_pred, self.target_expr)

        def train_results_str_fn(train_res):
            return '{} loss={:.6f}'.format(self.name, train_res[0])

        def eval_results_str_fn(eval_res):
            return '{} loss={:.6f}'.format(self.name, eval_res[0])

        return ObjectiveOutput(train_cost=train_loss_per_sample.mean() * self.cost_weight,
                               train_results=[train_loss_per_sample.sum()],
                               train_results_str_fn=train_results_str_fn,
                               eval_results=[eval_loss_per_sample.sum()],
                               eval_results_str_fn=eval_results_str_fn,
                               prediction=eval_pred)
