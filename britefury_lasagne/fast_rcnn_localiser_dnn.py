import theano.tensor as T

import lasagne

from . import basic_dnn, dnn_objective

class FastRCNNLocaliser (basic_dnn.BasicDNN):
    """
    A fast RCNN localiser DNN for object detection.
    """
    def __init__(self, input_vars, num_anchor_boxes, objectness_var, obj_mask_var, relbox_var, box_mask_var,
                 objectness_layer, relbox_layer, num_fg_classes=1, trainable_params=None, updates_fn=None,
                 params_source=None):
        """
        Predicts for N anchor boxes

        :param input_vars: input variables
        :param objectness_var: a `(sample,N,height,width)` binary as int32 variable indicating if the box contains
        an object of interest
        :param obj_mask_var: `None` or a `(sample,N,height,width)` binary as float32 variable indicating if the
        network should learn from the corresponding value in `objectness_var`
        :param relbox_var: a `(sample,4*N,height,width)` float32 variable that gives the box position and size relative
        to the corresponding anchor box
        :param box_mask_var: `None` or a a `(sample,N,height,width)` binary as float32 variable indicating if the
        network should learn from the corresponding values in `relbox_var`
        :param objectness_layer: a Lasagne layer that produces objectness classification predictions; should have
        N*2 channels and identity non-linearity
        :param relbox_layer: a Lasagne layer that produces relative box position and size predictions; should have
        N*4 channels and identity non-linearity
        :param num_fg_classes: the number of foreground classes
        :param trainable_params: [optional] parameters to train
        :param updates_fn: [optional] a function of the form `fn(cost, params)` that computes the update expressions
        required to update the state of the network
        :param params_source: [optional] a source from which to acquire network parameters; either a string that
        provides the path of a NumPy file to load or a `BasicDNN` instance or a Lasagne layer or sequence of
        Lasagne layers.
        """
        n_classes = num_fg_classes + 1
        # The `relbox_var` variable has 4 components, so expand the mask as appropriate
        if objectness_layer.output_shape[1] != num_anchor_boxes*n_classes:
            raise ValueError('objectness_layer should output N*{} ({}) channels, not {}'.format(
                n_classes, num_anchor_boxes*n_classes, objectness_layer.output_shape[1]
            ))
        if relbox_layer.output_shape[1] != num_anchor_boxes*4:
            raise ValueError('relbox_layer should output N*4 ({}) channels, not {}'.format(
                num_anchor_boxes*4, relbox_layer.output_shape[1]
            ))

        if box_mask_var is not None:
            box_mask_expr = T.tile(box_mask_var, (1,4,1,1))
        else:
            box_mask_expr = None

        self.num_anchor_boxes = num_anchor_boxes
        self.objectness_objectives = []
        for i in range(num_anchor_boxes):
            obj_i_expr = objectness_var[:,i*n_classes,:,:]
            obj_i_mask = obj_mask_var[:,i*n_classes,:,:] if obj_mask_var is not None else None

            obj_i_layer = lasagne.layers.SliceLayer(objectness_layer, indices=slice(i*n_classes, i*n_classes+n_classes), axis=1)
            obj_i_objective = dnn_objective.ClassifierObjective('objectness_{}'.format(i), obj_i_layer, obj_i_expr,
                                                                mask_expr=obj_i_mask, n_target_spatial_dims=2,
                                                                target_channel_index=None)
            self.objectness_objectives.append(obj_i_objective)
        self.relbox_objective = dnn_objective.RegressorObjective('relbox', relbox_layer, relbox_var, mask_expr=box_mask_expr,
                                                                 n_target_spatial_dims=2)

        target_and_mask_vars = [objectness_var]
        if obj_mask_var is not None:
            target_and_mask_vars.append(obj_mask_var)
        target_and_mask_vars.append(relbox_var)
        if box_mask_var is not None:
            target_and_mask_vars.append(box_mask_var)

        super(FastRCNNLocaliser, self).__init__(input_vars, target_and_mask_vars, [objectness_layer, relbox_layer],
                                                self.objectness_objectives + [self.relbox_objective],
                                                score_objective=self.objectness_objectives,
                                                trainable_params=trainable_params, updates_fn=updates_fn,
                                                params_source=params_source)

