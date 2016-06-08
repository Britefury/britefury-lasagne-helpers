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

def _flatten_spatial(x):
    ndims = len(x.shape)
    n_channels = x.shape[1]
    if ndims > 2:
        # (Sample,Channel,Spatial...) -> (Sample,Spatial...,Channel)
        x = np.rollaxis(x, 1, ndims)
        # (Sample,Spatial...,Channel) -> (Sample:Spatial...,Channel)
        x = x.reshape((-1, n_channels))
    return x

def _unflatten_spatial(x, spatial_shape, n_channels):
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



class AbstractClassifierObjective (AbstractObjective):
    def __init__(self, name, objective_layer, target_expr, cost_weight=1.0):
        super(AbstractClassifierObjective, self).__init__(name, cost_weight)
        self.objective_layer = objective_layer
        self.target_expr = target_expr


    def _build_classifier(self, prob_layer):
        # Get an expression representing the predicted probability
        train_pred_prob = lasagne.layers.get_output(prob_layer)

        # Create a per-sample loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        train_loss_per_sample = lasagne.objectives.categorical_crossentropy(train_pred_prob, self.target_expr)

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_pred_prob = lasagne.layers.get_output(prob_layer, deterministic=True)
        # Create evaluation loss expression
        eval_loss_per_sample = lasagne.objectives.categorical_crossentropy(eval_pred_prob, self.target_expr)
        # Create an expression for error count
        eval_err_count = T.sum(T.neq(T.argmax(eval_pred_prob, axis=1), self.target_expr),
                                     dtype=theano.config.floatX)

        def train_results_str_fn(train_res):
            return '{} loss={:.6f}'.format(self.name, train_res[0])

        def eval_results_str_fn(eval_res):
            return '{} loss={:.6f} err={:.2%}'.format(self.name, eval_res[0], eval_res[1])

        return ObjectiveOutput(train_cost=train_loss_per_sample.mean() * self.cost_weight,
                               train_results=[train_loss_per_sample.sum()],
                               train_results_str_fn=train_results_str_fn,
                               eval_results=[eval_loss_per_sample.sum(), eval_err_count],
                               eval_results_str_fn=eval_results_str_fn,
                               prediction=eval_pred_prob)


class SampleClassifierObjective (AbstractClassifierObjective):
    def build(self):
        softmax = TemperatureSoftmax()

        # Predicted probability layer
        prob_layer = lasagne.layers.NonlinearityLayer(self.objective_layer, softmax)

        return self._build_classifier(prob_layer)


class ConvolutionalClassifierObjective (AbstractClassifierObjective):
    def __init__(self, name, objective_layer, target_expr, cost_weight=1.0):
        super(ConvolutionalClassifierObjective, self).__init__(self, name, objective_layer, target_expr,
                                                               cost_weight)
        self._spatial_shape = None
        self._n_channels = None
    
    def build(self):
        softmax = TemperatureSoftmax()

        # Flatten the
        obj_flat_layer, self._spatial_shape, self._n_channels = \
            _flatten_spatial_lasagne_layer(self.objective_layer)

        # Predicted probability layer
        prob_layer = lasagne.layers.NonlinearityLayer(obj_flat_layer, softmax)

        return self._build_classifier(prob_layer)


    def transform_input_target(self, x):
        return _flatten_spatial(x)

    def inv_transform_prediction(self, x):
        return _unflatten_spatial(x, self._spatial_shape, self._n_channels)




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
