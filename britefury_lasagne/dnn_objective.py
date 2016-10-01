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
            x = x.flatten()
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
        """
        Abstract objective

        :param name: name of objective
        :param cost_weight: (default=1.0) weight applied to this objectives cost
        """
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

    def __init__(self, name, objective_layer, target_expr, mask_expr=None, n_target_spatial_dims=0,
                 target_channel_index=None, score=SCORE_ERROR, cost_weight=1.0):
        """
        Multi-class classifier objective

        :param name: objective name
        :param objective_layer: Lasagne layer that with a softmax non-linearity added to it will predict the class
        probabilities
        :param target_expr: ground truth target expression
        :param mask_expr: [optional] mask expression
        :param n_target_spatial_dims: (default=0) number of spatial dimensions for the target
        :param target_channel_index: (default=None) if the target has a channel dimension, this is the index into
        that channel for this objective
        :param score: (default=`'err'`) how to evaluate this objective; one of `'err'`, `'jaccard'`, `'precision'`,
        `'recall'`, `'f1'` or use the constants `ClassifierObjective.SCORE_ERROR`, `ClassifierObjective.SCORE_JACCARD`,
        `ClassifierObjective.SCORE_PRECISION`, `ClassifierObjective.SCORE_RECALL`, `ClassifierObjective.SCORE_F1`
        respectively
        :param cost_weight: (default=1.0) weight applied to the cost of this objective
        """
        super(ClassifierObjective, self).__init__(name, cost_weight)
        self.objective_layer = objective_layer
        self.target_expr = target_expr
        self.mask_expr = mask_expr
        self.n_target_spatial_dims = n_target_spatial_dims
        self.target_channel_index = target_channel_index
        self.score = score
        self.softmax = TemperatureSoftmax()


    @property
    def temperature(self):
        return self.softmax.temperature

    @temperature.setter
    def temperature(self, t):
        self.softmax.temperature = t


    def build(self):
        if self.target_channel_index is not None:
            target_expr = self.target_expr[:,self.target_channel_index]
            mask_expr = self.mask_expr[:,self.target_channel_index] if self.mask_expr is not None else None
        else:
            target_expr = self.target_expr
            mask_expr = self.mask_expr

        flat_target = _flatten_spatial_theano(target_expr, None)
        flat_mask = _flatten_spatial_theano(mask_expr, None) if self.mask_expr is not None else None

        # Flatten the objective layer (if the objective layer generates an
        # output with 2 dimensions then this is a no-op)
        obj_flat_layer, spatial_shape, n_classes = \
            _flatten_spatial_lasagne_layer(self.objective_layer)

        n_spatial = np.prod(spatial_shape)
        inv_n_spatial = lasagne.utils.floatX(1.0 / float(n_spatial))

        # Predicted probability layer
        prob_layer = lasagne.layers.NonlinearityLayer(obj_flat_layer, self.softmax)

        # Get an expression representing the predicted probability
        train_pred_prob = lasagne.layers.get_output(prob_layer)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        train_loss = lasagne.objectives.categorical_crossentropy(train_pred_prob, flat_target)
        if flat_mask is not None:
            train_loss = train_loss * flat_mask

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_pred_prob = lasagne.layers.get_output(prob_layer, deterministic=True)
        # Create evaluation loss expression
        eval_loss = lasagne.objectives.categorical_crossentropy(eval_pred_prob, flat_target)
        if flat_mask is not None:
            eval_loss = eval_loss * flat_mask
        # Predicted class
        eval_pred_cls = T.argmax(eval_pred_prob, axis=1)

        eval_scores = []

        if self.score == self.SCORE_ERROR:
            # Create an expression for error count
            errors = T.neq(eval_pred_cls, flat_target).astype(theano.config.floatX)
            if flat_mask is not None:
                errors = errors * flat_mask

            if self.n_target_spatial_dims == 0:
                eval_scores = [errors.sum()]
            else:
                eval_scores = [errors.sum() * inv_n_spatial]
        elif self.score in {self.SCORE_JACCARD, self.SCORE_PRECISION, self.SCORE_RECALL, self.SCORE_F1}:
            for cls_i in range(n_classes):
                truth = T.eq(flat_target, cls_i).astype(theano.config.floatX)
                pred = T.eq(eval_pred_cls, cls_i).astype(theano.config.floatX)
                inv_truth = T.neq(flat_target, cls_i).astype(theano.config.floatX)
                inv_pred = T.neq(eval_pred_cls, cls_i).astype(theano.config.floatX)
                true_neg = inv_truth * inv_pred
                true_pos = truth * pred
                false_neg = truth * inv_pred
                false_pos = inv_truth * pred
                if flat_mask is not None:
                    true_neg = true_neg * flat_mask
                    true_pos = true_pos * flat_mask
                    false_neg = false_neg * flat_mask
                    false_pos = false_pos * flat_mask

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
            return '{} loss={:.6f} {}={:.2%}'.format(self.name, eval_res[0], self.score, self._compute_score(eval_res))

        return ObjectiveOutput(train_cost=train_loss.mean() * self.cost_weight,
                               train_results=[train_loss_batch],
                               train_results_str_fn=train_results_str_fn,
                               eval_results=[eval_loss_batch] + eval_scores,
                               eval_results_str_fn=eval_results_str_fn,
                               prediction=pred_prob)

    def _score_frac(self, numerator, denominator):
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator

    def _compute_score(self, eval_results):
        if self.score == self.SCORE_ERROR:
            return eval_results[1]
        elif self.score in {self.SCORE_JACCARD, self.SCORE_PRECISION, self.SCORE_RECALL, self.SCORE_F1}:
            components = eval_results[1:]
            cls_scores = []
            for cls_i in range(len(components) // 4):
                true_neg = components[cls_i * 4 + 0]
                false_neg = components[cls_i * 4 + 1]
                false_pos = components[cls_i * 4 + 2]
                true_pos = components[cls_i * 4 + 3]

                cls_n = false_neg + false_pos + true_pos

                if cls_n > 0:
                    if self.score == self.SCORE_JACCARD:
                        cls_scores.append(self._score_frac(true_pos, false_pos + false_neg + true_pos))
                    elif self.score == self.SCORE_PRECISION:
                        cls_scores.append(self._score_frac(true_pos, false_pos + true_pos))
                    elif self.score == self.SCORE_RECALL:
                        cls_scores.append(self._score_frac(true_pos, false_neg + true_pos))
                    elif self.score == self.SCORE_F1:
                        precision = self._score_frac(true_pos, false_pos + true_pos)
                        recall = self._score_frac(true_pos, false_neg + true_pos)
                        f1 = self._score_frac(2.0 * (precision * recall), precision + recall)
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
    def __init__(self, name, objective_layer, target_expr, mask_expr=None, n_target_spatial_dims=0,
                 cost_weight=1.0):
        """
        Regression objective

        :param name: objective name
        :param objective_layer: Lasagne layer that will predict the output
        :param target_expr: ground truth target expression
        :param mask_expr: [optional] mask expression
        :param n_target_spatial_dims: (default=0) number of spatial dimensions for the target
        :param cost_weight: (default=1.0) weight applied to the cost of this objective
        """
        super(RegressorObjective, self).__init__(name, cost_weight)
        self.objective_layer = objective_layer
        self.target_expr = target_expr
        self.mask_expr = mask_expr
        self.n_target_spatial_dims = n_target_spatial_dims


    def build(self):
        # Get an expression representing the predicted probability
        train_pred = lasagne.layers.get_output(self.objective_layer)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize; squared error
        train_loss = lasagne.objectives.squared_error(train_pred, self.target_expr)
        if self.mask_expr is not None:
            train_loss = train_loss * self.mask_expr

        # Create prediction expressions; use deterministic forward pass (disable
        # dropout layers)
        eval_pred = lasagne.layers.get_output(self.objective_layer, deterministic=True)
        # Create evaluation loss expression
        eval_loss = lasagne.objectives.squared_error(eval_pred, self.target_expr)
        if self.mask_expr is not None:
            eval_loss = eval_loss * self.mask_expr

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
