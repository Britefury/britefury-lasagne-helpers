import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.utils import floatX


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

    def score_improved(self, new_results, best_so_far_results):
        raise NotImplementedError('Abstract for {}'.format(type(self)))



class ClassifierObjective (AbstractObjective):
    SCORE_ERROR = 'err'
    SCORE_JACCARD = 'jaccard'
    SCORE_PRECISION = 'precision'
    SCORE_RECALL = 'recall'
    SCORE_F1 = 'f1'

    def __init__(self, name, objective_layer, target_expr, n_target_spatial_dims, score=SCORE_ERROR, cost_weight=1.0):
        super(ClassifierObjective, self).__init__(name, cost_weight)
        self.objective_layer = objective_layer
        self.target_expr = target_expr
        self.n_target_spatial_dims = n_target_spatial_dims
        self.score = score


    def build(self):
        flat_target = _flatten_spatial_theano(self.target_expr, None)
        if flat_target.ndim != 1:
            raise ValueError('target must have 1 dimensions, not {}'.format(flat_target.ndim))

        # Flatten the objective layer (if the objective layer generates an
        # output with 2 dimensions then this is a no-op)
        obj_flat_layer, spatial_shape, n_classes = \
            _flatten_spatial_lasagne_layer(self.objective_layer)

        n_spatial = np.prod(spatial_shape)
        inv_n_spatial = lasagne.utils.floatX(1.0 / float(n_spatial))

        # Predicted probability layer
        softmax = TemperatureSoftmax()
        prob_layer = lasagne.layers.NonlinearityLayer(obj_flat_layer, softmax)

        # Get an expression representing the predicted probability
        train_pred_prob = lasagne.layers.get_output(prob_layer)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        train_loss = lasagne.objectives.categorical_crossentropy(train_pred_prob, flat_target)

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_pred_prob = lasagne.layers.get_output(prob_layer, deterministic=True)
        # Create evaluation loss expression
        eval_loss = lasagne.objectives.categorical_crossentropy(eval_pred_prob, flat_target)
        # Predicted class
        eval_pred_cls = T.argmax(eval_pred_prob, axis=1)

        eval_scores = []

        if self.score == self.SCORE_ERROR:
            # Create an expression for error count
            errors = T.neq(eval_pred_cls, flat_target).astype(theano.config.floatX)

            if self.n_target_spatial_dims == 0:
                eval_scores = [errors.sum()]
            else:
                eval_scores = [errors.sum() * inv_n_spatial]
        elif self.score in {self.SCORE_JACCARD, self.SCORE_PRECISION, self.SCORE_RECALL, self.SCORE_F1}:
            for cls_i in range(n_classes):
                truth = T.eq(flat_target, cls_i).astype(theano.config.floatX)
                pred = T.eq(eval_pred_cls, cls_i).astype(theano.config.floatX)
                n_truth = T.neq(flat_target, cls_i).astype(theano.config.floatX)
                n_pred = T.neq(eval_pred_cls, cls_i).astype(theano.config.floatX)
                true_neg = n_truth * n_pred
                true_pos = truth * pred
                false_neg = truth * n_pred
                false_pos = n_truth * pred

                eval_scores.extend([true_neg.sum(), false_neg.sum(), false_pos.sum(), true_pos.sum()])
        else:
            raise ValueError('score is not valid ({})'.format(self.score))

        if self.n_target_spatial_dims == 0:
            train_loss_batch = train_loss.sum()
            eval_loss_batch = eval_loss.sum()
        else:
            train_loss_batch = train_loss.sum() * inv_n_spatial
            eval_loss_batch = eval_loss.sum() * inv_n_spatial

        # Unflatten prediction
        pred_prob = _unflatten_spatial_theano(eval_pred_prob, spatial_shape, n_classes)

        def train_results_str_fn(train_res):
            return '{} loss={:.6f}'.format(self.name, train_res[0])

        def eval_results_str_fn(eval_res):
            # return '{} loss={:.6f} {}={:.2%}'.format(self.name, eval_res[0], self.score, eval_res[1])
            return '{} loss={:.6f} {}={:.2%}'.format(self.name, eval_res[0], self.score, self._compute_score(eval_res))
            # return '{} loss={:.6f} {}={}'.format(self.name, eval_res[0], self.score, eval_res[1:])

        return ObjectiveOutput(train_cost=train_loss.mean() * self.cost_weight,
                               train_results=[train_loss_batch],
                               train_results_str_fn=train_results_str_fn,
                               eval_results=[eval_loss_batch] + eval_scores,
                               eval_results_str_fn=eval_results_str_fn,
                               prediction=pred_prob)

    def _compute_score(self, eval_results):
        components = eval_results[1:]
        cls_scores = []
        for cls_i in range(len(components) // 4):
            true_neg = components[cls_i * 4 + 0]
            false_neg = components[cls_i * 4 + 1]
            false_pos = components[cls_i * 4 + 2]
            true_pos = components[cls_i * 4 + 3]

            if self.score == self.SCORE_JACCARD:
                cls_scores.append(true_pos / (false_pos + false_neg + true_pos))
            elif self.score == self.SCORE_PRECISION:
                cls_scores.append(true_pos / (false_pos + true_pos))
            elif self.score == self.SCORE_RECALL:
                cls_scores.append(true_pos / (false_neg + true_pos))
            elif self.score == self.SCORE_F1:
                precision = true_pos / (false_pos + true_pos)
                recall = true_pos / (false_neg + true_pos)
                f1 = 2.0 * (precision * recall) / max(precision + recall, 1.0)
                cls_scores.append(f1)
            else:
                raise ValueError('score is not valid ({})'.format(self.score))
        return np.mean(cls_scores)



    def score_improved(self, new_results, best_so_far_results):
        if self.score == self.SCORE_ERROR:
            return new_results[1] < best_so_far_results[1]
        else:
            return self._compute_score(new_results) > self._compute_score(best_so_far_results)



class RegressorObjective (AbstractObjective):
    def __init__(self, name, objective_layer, target_expr, n_target_spatial_dims, cost_weight=1.0):
        super(RegressorObjective, self).__init__(name, cost_weight)
        self.objective_layer = objective_layer
        self.target_expr = target_expr
        self.n_target_spatial_dims = n_target_spatial_dims


    def build(self):
        # Get an expression representing the predicted probability
        train_pred = lasagne.layers.get_output(self.objective_layer)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize; squared error
        train_loss = lasagne.objectives.squared_error(train_pred, self.target_expr)

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_pred = lasagne.layers.get_output(self.objective_layer, deterministic=True)
        # Create evaluation loss expression
        eval_loss = lasagne.objectives.squared_error(eval_pred, self.target_expr)

        if self.n_target_spatial_dims == 0:
            train_loss_batch = train_loss.sum()
            eval_loss_batch = eval_loss.sum()
        else:
            train_loss_batch = train_loss.mean(axis=tuple(range(2,2+self.n_target_spatial_dims))).sum()
            eval_loss_batch = eval_loss.mean(axis=tuple(range(2,2+self.n_target_spatial_dims))).sum()

        def train_results_str_fn(train_res):
            return '{} loss={:.6f}'.format(self.name, train_res[0])

        def eval_results_str_fn(eval_res):
            return '{} loss={:.6f}'.format(self.name, eval_res[0])

        return ObjectiveOutput(train_cost=train_loss.mean() * self.cost_weight,
                               train_results=[train_loss_batch],
                               train_results_str_fn=train_results_str_fn,
                               eval_results=[eval_loss_batch],
                               eval_results_str_fn=eval_results_str_fn,
                               prediction=eval_pred)

    def score_improved(self, new_results, best_so_far_results):
        return new_results[0] < best_so_far_results[0]
