from __future__ import print_function

import numpy as np


def _box_i_over_u(a_y, a_x, b_y, b_x):
    """
    Compute intersection over union of box
    :param a_y: tuple `(y_start, y_end)` of box A
    :param a_x: tuple `(x_start, x_end)` of box A
    :param b_y: tuple `(y_start, y_end)` of box B
    :param b_x: tuple `(x_start, x_end)` of box B
    :return: area of intersection / area of union
    """
    i_l_y = max(a_y[0], b_y[0])
    i_u_y = min(a_y[1], b_y[1])
    i_l_x = max(a_x[0], b_x[0])
    i_u_x = min(a_x[1], b_x[1])
    i_h = i_u_y - i_l_y
    i_w = i_u_x - i_l_x
    i_area = max(i_h, 0) * max(i_w, 0)
    a_area = (a_y[1] - a_y[0]) * (a_x[1] - a_x[0])
    b_area = (b_y[1] - b_y[0]) * (b_x[1] - b_x[0])
    return float(i_area) / float(a_area + b_area - i_area)


def _rel_box(target, anchor):
    """
    Compute target box relative to anchor box, where centre is offset normalized to anchor size
    and size is log of ratio of target to anchor size.

    Note that the relative centre is in the range [-1, 1] assuming that the target centre lies within
    the bounds of the anchor box.

    :param target: `(centre_y, centre_x, height, width)` of target box
    :param anchor: `(centre_y, centre_x, height, width)` of anchor box
    :return: numpy array of the form `[rel_centre_y, rel_centre_x, log_height_ratio, log_width_ratio]`
    """
    return np.array([
        float((target[0] - anchor[0]) * 2.0) / anchor[2],
        float((target[1] - anchor[1]) * 2.0) / anchor[3],
        np.log(float(target[2]) / anchor[2]),
        np.log(float(target[3]) / anchor[3]),
    ])


def bbox_to_centre_size_box(bbox):
    """
    Convert a bounding box (or array of bounding boxes) of the form (lower_y, lower_x, upper_y, upper_x)
    to a centre-size form (centre_y, centre_x, height, width)
    :param bbox: NumPy array with the size of the last dimension being 4
    :return: NumPy array of the same shape as `bbox`
    """
    lower_y = bbox[..., 0:1]
    lower_x = bbox[..., 1:2]
    upper_y = bbox[..., 2:3]
    upper_x = bbox[..., 3:4]
    centre_y = (lower_y + upper_y) * 0.5
    centre_x = (lower_x + upper_x) * 0.5
    height = upper_y - lower_y
    width = upper_x - lower_x
    return np.concatenate([centre_y, centre_x, height, width], axis=bbox.ndim - 1)


def centre_size_box_to_bbox(box):
    """
    Convert a centre-size box (or array of boxes) of the form form (centre_y, centre_x, height, width) to
    a bounding box of the form (lower_y, lower_x, upper_y, upper_x)
    :param box: NumPy array with the size of the last dimension being 4
    :return: NumPy array of the same shape as `box`
    """
    centre_y = box[..., 0:1]
    centre_x = box[..., 1:2]
    height = box[..., 2:3]
    width = box[..., 3:4]
    lower_y = centre_y - height * 0.5
    lower_x = centre_x - width * 0.5
    upper_y = centre_y + height * 0.5
    upper_x = centre_x + width * 0.5
    return np.concatenate([lower_y, lower_x, upper_y, upper_x], axis=box.ndim - 1)


def get_anchor_boxes_overlapping(anchor_grid_origin, anchor_grid_cell_size, anchor_grid_shape, anchor_box_sizes,
                                 bbox):
    """
    Get the anchor boxes that overlap the box `bbox`

    :param anchor_grid_origin: the position of the centre of the anchor at grid point 0,0
    as a (2,) array of the form (centre_y, centre_x)
    :param anchor_grid_cell_size: the offset between centres of anchor points as a (2,) array
    of the form (delta_y, delta_x)
    :param anchor_grid_shape: the size of the grid of anchors of the form (n_y, n_x)
    :param anchor_box_sizes: the list of anchor box sizes in the form of a (N,2) array, where
    each row is of the form (height, width)
    :param bbox: the box against which the anchor boxes are to be tested as (4,) array of the form
    (lower_y, lower_x, upper_y, upper_x)

    :return: an (N,2,2) array where each of the N entries corresponds to a box size in `anchor_box_sizes`
    and each (2,2) entry is of the form `[[y_start,y_end], [x_start, x_end]]`; the array shape is:
    `(anchor_box_index, axis, start_or_end)`.
    """
    centre = (bbox[:2] + bbox[2:]) * 0.5
    size = bbox[2:] - bbox[:2]
    # Get the offset from the origin of the anchor grid
    box_pos_in_grid = centre - anchor_grid_origin
    # Compute the overlap bounds; half the sum of the box sizes
    overlap_bounds = (anchor_box_sizes + size[None, :]) * 0.5
    # Compute the start grid row and column for each anchor box size
    start = np.ceil((box_pos_in_grid[None,:] - overlap_bounds) / anchor_grid_cell_size).astype(int)
    # Compute the end grid row and column for each anchor box size
    end = np.floor((box_pos_in_grid[None,:] + overlap_bounds) / anchor_grid_cell_size).astype(int) + 1
    return np.append(start[:,:,None], end[:,:,None], axis=2)


def get_anchor_boxes_in_range(anchor_grid_origin, anchor_grid_cell_size, anchor_grid_shape, anchor_box_sizes,
                              img_shape):
    """
    Get the anchor boxes that completely inside the rectangle [0,0] -> `img_shape`

    :param anchor_grid_origin: the position of the centre of the anchor at grid point 0,0
    as a (2,) array of the form (centre_y, centre_x)
    :param anchor_grid_cell_size: the offset between centres of anchor points as a (2,) array
    of the form (delta_y, delta_x)
    :param anchor_grid_shape: the size of the grid of anchors of the form (n_y, n_x)
    :param anchor_box_sizes: the list of anchor box sizes in the form of a (N,2) array, where
    each row is of the form (height, width)
    :param img_shape: a numpy array `[height, width]` that defines the bounds of the image

    :return: an (N,2,2) array where each of the N entries corresponds to a box size in `anchor_box_sizes`
    and each (2,2) entry is of the form `[[y_start,y_end], [x_start, x_end]]`; the array shape is:
    `(anchor_box_index, axis, start_or_end)`.
    """
    # Get the offset from the origin of the anchor grid
    box_pos_in_grid = img_shape * 0.5 - anchor_grid_origin
    # Compute the overlap bounds; half the sum of the box sizes
    overlap_bounds = (img_shape[None,:] - anchor_box_sizes) * 0.5
    # Compute the start grid row and column for each anchor box size
    start = np.ceil((box_pos_in_grid[None,:] - overlap_bounds) / anchor_grid_cell_size).astype(int)
    # Compute the end grid row and column for each anchor box size
    end = np.floor((box_pos_in_grid[None,:] + overlap_bounds) / anchor_grid_cell_size).astype(int) + 1
    return np.append(start[:,:,None], end[:,:,None], axis=2)


def compute_coverage(anchor_grid_origin, anchor_grid_cell_size, anchor_grid_shape, anchor_range, anchor_box_size,
                     bbox_lower, bbox_upper, bbox_area):
    """
    Compute the intersection over union overlap fraction between the bounding box defined by `bbox_lower`
    and `bbox_upper` and all the anchor boxes with the size `anchor_box_size` whose centres lie on a grid
    whose origin is at `anchor_grid_origin` with cell size is `anchor_grid_cell_size` and grid shape (number
    of cells) is `anchor_grid_shape`, only checking the cells within the range specified by `anchor_range`

    :param anchor_grid_origin: the origin of the anchor box grid (a `(2,)` array)
    :param anchor_grid_cell_size: the spacing between anchor box centres (a `(2,)` array)
    :param anchor_grid_shape: the size of the anchor box grid (number of cells) (a `(2,)` array)
    :param anchor_range: the range to check for intersection with the bounding box being tested
    (a `(2,2)` array of the form `[[y_start,y_end],[x_start,x_end]]`)
    :param anchor_box_size: the size of the anchor box (a `(2,)` array)
    :param bbox_lower: the lower corner of the bounding box being tested (a `(2,)` array)
    :param bbox_upper: the upper corner of the bounding box being tested (a `(2,)` array)
    :param bbox_area: the area of the bounding box being tested (passed as a parameter as an optimisation)
    :return: a `(M,N)` array where `M == anchor_range[0,1]-anchor_range[0,0]` and
    `N == anchor_range[1,1]-anchor_range[1,0]`
    """
    ystart = anchor_range[0,0]
    yend = anchor_range[0,1]
    xstart = anchor_range[1,0]
    xend = anchor_range[1,1]
    # Get the dimensions of this anchor
    h = anchor_box_size[0]
    w = anchor_box_size[1]
    # Compute the y and x coords of the centres of the anchor boxes
    anchor_cen_y = anchor_grid_origin[0] + np.arange(ystart, yend) * anchor_grid_cell_size[0]
    anchor_cen_x = anchor_grid_origin[1] + np.arange(xstart, xend) * anchor_grid_cell_size[1]
    # Compute the lower and upper bounds in y and x of the anchor boxes
    anchor_cen_l_y = anchor_cen_y - h * 0.5
    anchor_cen_u_y = anchor_cen_y + h * 0.5
    anchor_cen_l_x = anchor_cen_x - w * 0.5
    anchor_cen_u_x = anchor_cen_x + w * 0.5

    # Compute the amount of overlap in y and x between `gt_box` and the anchor boxes it potentially intersects
    intersection_y = np.minimum(anchor_cen_u_y, bbox_upper[0]) - np.maximum(anchor_cen_l_y, bbox_lower[0])
    intersection_x = np.minimum(anchor_cen_u_x, bbox_upper[1]) - np.maximum(anchor_cen_l_x, bbox_lower[1])
    intersection_y = np.maximum(intersection_y, 0.0)
    intersection_x = np.maximum(intersection_x, 0.0)
    intersection_area = intersection_y[:,None] * intersection_x[None,:]
    return intersection_area / (h * w + bbox_area - intersection_area)


def ground_truth_boxes_to_y(anchor_grid_origin, anchor_grid_cell_size, anchor_grid_shape, anchor_box_sizes,
                            img_shape, ground_truth_bboxes, box_classes=None, coverage_lower_threshold=0.3,
                            coverage_upper_threshold=0.7):
    """
    Transform a list of ground truth boxes into training targets for a localiser RCNN.

    :param anchor_grid_origin: the position of the centre of the anchor at grid point 0,0
    as a `(2,)` array of the form `(centre_y, centre_x)`
    :param anchor_grid_cell_size: the offset between centres of anchor points as a `(2,)` array
    of the form `(delta_y, delta_x)`
    :param anchor_grid_shape: the size of the grid of anchors of the form `(n_y, n_x)`
    :param anchor_box_sizes: the list of anchor box sizes in the form of a `(N,2)` array, where
    each row is of the form `(height, width)`
    :param img_shape: The shape of the image as a `(2,)` array of the form `(height, width)`
    :param ground_truth_bboxes: the list of ground-truth bounding boxes as a `(M,4)` array, where each row
    is of the form `(lower_y, lower_x, upper_y, upper_x)`
    :param box_classes: [optional] the list of ground-truth box classes as a `(M,)` integer array, where each
    element is the class index of the corresponding box; note that class indices must be >= 1, as 0 is reserved
    as the background class
    :param coverage_lower_threshold: if the intersection-over-union coverage of an anchor box by a ground truth
    box is less than or equal to this value then the box is not said to contain an object; 'objectness'
    classification of 0
    :param coverage_upper_threshold: if the intersection-over-union coverage of an anchor box by a ground truth
    box is greater than or equal to this value then the box is said to contain an object; 'objectness'
    classification of 1
    :return: tuple of `(y_objectness, y_obj_mask, y_rel_boxes, y_boxes_mask)` with the following shapes:
    `y_objectness` - `(N,S,T)` where `S,T = anchor_grid_shape` and `N ==  anchor_box_sizes.shape[0]`
    `y_obj_mask` - `(N,S,T)`
    `y_rel_boxes` - `(4,N,S,T)`
    `y_boxes_mask` - are `(N,S,T)`
    `S` and `T` are the anchor grid dimensions, N is the number of anchor boxes.
    Indices in the `N,S,T` dimensions uniquely identify an anchor box by position and size (determined by index)
    `y_objectness` is an integer array that indicates if an anchor box contains an object.
    `y_obj_mask` is a float32 array that indicates if there is learnable data in the corresponding entry
    of `y_objectness`; this is so if the coverage value is sufficiently 'definite' for this
    example to be used in training; if the coverage of an anchor box is between the two
    thresholds `coverage_lower_threshold` and `coverage_upper_threshold` then the 'obj_mask' classification is 0,
    1 otherwise
    `y_rel_boxes` is a float32 array that indicates the size of a ground truth box relative to the anchor;
    it takes the form of `(rel_y, rel_x, rel_h, rel_w)` where `rel_y` and `rel_x` is 2x the position of the ground
    truth box relative to that of the anchor box divided by the anchor boxes' height and width respectively
    (e.g. `rel_y = (abs_y - anchor_centre_y) * 2 / anchor_height`).
    `rel_h` and `rel_w` are the log of the ratios of the sizes of the ground truth box and the anchor box;
    `rel_h == log(gt_box_h / anchor_box_h)`.
    `y_boxes_mask` is a float32 array that indicates if there is learnable data in the corresponding entry
    of `y_rel_boxes`; this is so if the coverage of an anchor box is above the threshold
    `coverage_upper_threshold`.
    """
    anchor_grid_origin = np.array(anchor_grid_origin, dtype=float)
    anchor_grid_cell_size = np.array(anchor_grid_cell_size, dtype=float)
    anchor_box_sizes = np.array(anchor_box_sizes, dtype=float)
    img_shape = np.array(img_shape, dtype=int).astype(float)
    ground_truth_bboxes = np.array(ground_truth_bboxes, dtype=float)

    if anchor_grid_origin.shape != (2,):
        raise ValueError('anchor_grid_origin must have shape (2,), not {}'.format(anchor_grid_origin.shape))
    if anchor_grid_cell_size.shape != (2,):
        raise ValueError('anchor_grid_cell_size must have shape (2,), not {}'.format(anchor_grid_cell_size.shape))
    if not isinstance(anchor_grid_shape, tuple):
        raise TypeError('anchor_grid_shape must be a tuple, not a {}'.format(type(anchor_grid_shape)))
    if len(anchor_grid_shape) != 2:
        raise ValueError('anchor_grid_shape must have length 2, not {}'.format(len(anchor_grid_shape)))
    if len(anchor_box_sizes.shape) != 2:
        raise ValueError('anchor_box_sizes must have 2 dimensions, not {}'.format(len(anchor_box_sizes.shape)))
    if anchor_box_sizes.shape[1] != 2:
        raise ValueError('anchor_box_sizes must have shape (N,2), not {}'.format(anchor_box_sizes.shape))
    if img_shape.shape != (2,):
        raise ValueError('img_shape must have shape (2,), not {}'.format(img_shape.shape))
    if len(ground_truth_bboxes.shape) != 2:
        raise ValueError('ground_truth_boxes must have 2 dimensions, not {}'.format(len(ground_truth_bboxes.shape)))
    if ground_truth_bboxes.shape[1] != 4:
        raise ValueError('ground_truth_boxes must have shape (N,4), not {}'.format(ground_truth_bboxes.shape))
    if box_classes is not None:
        box_classes = np.array(box_classes, dtype=np.int32)
        if box_classes.min() < 1:
            raise ValueError('All box classes should be >= 1')

    n_anchors = anchor_box_sizes.shape[0]
    # y_shape =
    y_objectness = np.zeros((n_anchors,) + anchor_grid_shape, dtype=np.int32)
    y_obj_mask = np.zeros((n_anchors,) + anchor_grid_shape, dtype=np.float32)
    y_coverage = np.zeros((n_anchors,) + anchor_grid_shape, dtype=float)
    y_rel_boxes = np.zeros((4, n_anchors) + anchor_grid_shape, dtype=np.float32)
    y_boxes_mask = np.zeros((n_anchors,) + anchor_grid_shape, dtype=np.float32)

    valid_ranges = get_anchor_boxes_in_range(anchor_grid_origin, anchor_grid_cell_size,
                                             anchor_grid_shape, anchor_box_sizes, img_shape)

    gt_lower = ground_truth_bboxes[:, :2]
    gt_upper = ground_truth_bboxes[:, 2:]
    gt_centre = (gt_lower + gt_upper) * 0.5
    gt_size = gt_upper - gt_lower
    gt_area = gt_size[:, 0] * gt_size[:, 1]

    for gt_box_i in range(ground_truth_bboxes.shape[0]):
        anchor_ranges = get_anchor_boxes_overlapping(anchor_grid_origin, anchor_grid_cell_size,
                                                     anchor_grid_shape, anchor_box_sizes,
                                                     ground_truth_bboxes[gt_box_i, :])
        anchor_ranges[:,:,0] = np.maximum(anchor_ranges[:,:,0], valid_ranges[:,:,0])
        anchor_ranges[:,:,1] = np.minimum(anchor_ranges[:,:,1], valid_ranges[:,:,1])
        anchor_ranges[:,0,1] = np.minimum(anchor_ranges[:,0,1], anchor_grid_shape[0] - 1)
        anchor_ranges[:,1,1] = np.minimum(anchor_ranges[:,1,1], anchor_grid_shape[1] - 1)

        for anchor_i in range(anchor_ranges.shape[0]):
            # Get the rectangular range of anchors for this size that potentially interesect with the bounding
            # box `gt_box`
            anchor_range = anchor_ranges[anchor_i,:,:]
            ystart = anchor_range[0,0]
            yend = anchor_range[0,1]
            xstart = anchor_range[1,0]
            xend = anchor_range[1,1]

            assert ystart >= 0
            assert yend < anchor_grid_shape[0]
            assert xstart >= 0
            assert xend < anchor_grid_shape[1]

            if yend >= ystart and xend >= xstart:
                # Compute the y and x coords of the centres of the anchor boxes
                anchor_cen_y = anchor_grid_origin[0] + np.arange(ystart, yend) * anchor_grid_cell_size[0]
                anchor_cen_x = anchor_grid_origin[1] + np.arange(xstart, xend) * anchor_grid_cell_size[1]

                # Get the dimensions of this anchor
                h = anchor_box_sizes[anchor_i, 0]
                w = anchor_box_sizes[anchor_i, 1]

                i_over_u = compute_coverage(anchor_grid_origin, anchor_grid_cell_size, anchor_grid_shape, anchor_range,
                                            anchor_box_sizes[anchor_i, :],
                                            gt_lower[gt_box_i], gt_upper[gt_box_i], gt_area[gt_box_i])

                improve = (i_over_u > y_coverage[anchor_i, ystart:yend, xstart:xend]) & \
                          (i_over_u > coverage_upper_threshold)

                box_rel_y = (gt_centre[gt_box_i, 0] - anchor_cen_y) * 2.0 / h
                box_rel_x = (gt_centre[gt_box_i, 1] - anchor_cen_x) * 2.0 / w
                box_rel_h = np.log(gt_size[gt_box_i, 0] / h)
                box_rel_w = np.log(gt_size[gt_box_i, 1] / w)

                y_rel_boxes[0, anchor_i, ystart:yend, xstart:xend][improve] = box_rel_y[:,None].repeat(xend-xstart,
                                                                                                       axis=1)[improve]
                y_rel_boxes[1, anchor_i, ystart:yend, xstart:xend][improve] = box_rel_x[None,:].repeat(yend-ystart,
                                                                                                       axis=0)[improve]
                y_rel_boxes[2, anchor_i, ystart:yend, xstart:xend][improve] = box_rel_h
                y_rel_boxes[3, anchor_i, ystart:yend, xstart:xend][improve] = box_rel_w

                y_coverage[anchor_i, ystart:yend, xstart:xend] = np.maximum(y_coverage[anchor_i, ystart:yend,
                                                                            xstart:xend], i_over_u)

                if box_classes is None:
                    y_objectness[anchor_i, ystart:yend, xstart:xend][improve] = 1
                else:
                    y_objectness[anchor_i, ystart:yend, xstart:xend][improve] = box_classes[gt_box_i]

    for anchor_i in range(anchor_ranges.shape[0]):
        val = valid_ranges[anchor_i]
        ((ys,ye), (xs,xe)) = val
        y_obj_mask[anchor_i,ys:ye,xs:xe][y_coverage[anchor_i,ys:ye,xs:xe] <= coverage_lower_threshold] = 1
        y_obj_mask[anchor_i,ys:ye,xs:xe][y_coverage[anchor_i,ys:ye,xs:xe] >= coverage_upper_threshold] = 1
        y_boxes_mask[anchor_i,ys:ye,xs:xe][y_coverage[anchor_i,ys:ye,xs:xe] >= coverage_upper_threshold] = 1

    return y_objectness, y_obj_mask, y_rel_boxes, y_boxes_mask

def y_to_boxes(y_obj_class, y_obj_confidence, y_relboxes, anchor_grid_origin, anchor_grid_cell_size, anchor_grid_shape,
               anchor_box_sizes):
    """
    Transform training targets - either ground truths or predictions - into predicted bounding boxes.

    :param y_obj_class: `None` or an integer array of shape `(N,S,T)` where
    `S,T = anchor_grid_shape` and `N ==  anchor_box_sizes.shape[0]` that give the predicted classes of the bounding
    boxes; if there is only 1 class, `0` should represent background or no object present and `1` should represent
    foreground or object present. If `y_obj_class` is `None`, `None` will be returned for `box_classes`.
    :param y_obj_confidence: None or a float or a float array of shape `(N,S,T)` where
    `S,T = anchor_grid_shape` and `N ==  anchor_box_sizes.shape[0]` that give the confidence of the class predictions
    in `y_obj_class`. If `None`, a value of `1.0` is used for all confidence values, if a float is used, it is
    used as a constant confience for all class predictions.
    :param y_relboxes: a float32 array of shape `(4,N,S,T)` that indicates the position and size of a box
    relative to the cirresponding anchor (see `anchor_box_sizes` parameter); it takes the form of
    `(rel_y, rel_x, rel_h, rel_w)` where `rel_y` and `rel_x` is the position of the ground truth box relative to that
    of the anchor box divided by the anchor boxes' height and width respectively
    (e.g. `rel_y = (abs_y - anchor_centre_y) * 2 / anchor_height`). `rel_h` and `rel_w` are the log of
    the ratios of the sizes of the ground truth box and the anchor box; `rel_h == log(gt_box_h / anchor_box_h)`.
    :param anchor_grid_origin: the position of the centre of the anchor at grid point 0,0
    as a `(2,)` array of the form `(centre_y, centre_x)`
    :param anchor_grid_cell_size: the offset between centres of anchor points as a `(2,)` array
    of the form `(delta_y, delta_x)`
    :param anchor_grid_shape: the size of the grid of anchors of the form `(n_y, n_x)`
    :param anchor_box_sizes: the list of anchor box sizes in the form of a `(N,2)` array, where
    each row is of the form `(height, width)`
    :return: tuple of (boxes, box_classes, box_conf), where:
    `bboxes` is a list of boxes as a `(M,4)` array, where each row is of the form
    `(lower_y, lower_x, upper_y, upper_x)`
    `box_classes` is a `(M,)` array where each element is the predicted class of the box, or `None` if
        `y_obj_class` is None
    `box_conf` is a `(M,)` array where each element is confidence of the box class prediction
    """
    if y_obj_class is None:
        n_anchors, n_y, n_x = y_obj_confidence.shape

        y = np.arange(n_y) * anchor_grid_cell_size[0] + anchor_grid_origin[0]
        x = np.arange(n_x) * anchor_grid_cell_size[1] + anchor_grid_origin[1]

        cen_y = (y_relboxes[2, :, :, :] * anchor_box_sizes[:, 0, None, None] * 0.5 + y[None, :, None]).flatten()
        cen_x = (y_relboxes[3, :, :, :] * anchor_box_sizes[:, 1, None, None] * 0.5 + x[None, None, :]).flatten()
        h = (np.exp(y_relboxes[0, :, :, :]) * anchor_box_sizes[:, 0, None, None]).flatten()
        w = (np.exp(y_relboxes[1, :, :, :]) * anchor_box_sizes[:, 1, None, None]).flatten()
    else:
        box_mask = y_obj_class != 0
        ndx = np.indices(box_mask.shape).reshape((3, -1))[:, box_mask.flatten()].T

        anchor_i = ndx[:, 0]
        cell_y = ndx[:, 1]
        cell_x = ndx[:, 2]
        y = cell_y.astype(float) * anchor_grid_cell_size[0] + anchor_grid_origin[0]
        x = cell_x.astype(float) * anchor_grid_cell_size[1] + anchor_grid_origin[1]

        cen_y = y_relboxes[2, anchor_i, cell_y, cell_x] * anchor_box_sizes[anchor_i, 0] * 0.5 + y
        cen_x = y_relboxes[3, anchor_i, cell_y, cell_x] * anchor_box_sizes[anchor_i, 1] * 0.5 + x
        h = np.exp(y_relboxes[0, anchor_i, cell_y, cell_x]) * anchor_box_sizes[anchor_i, 0]
        w = np.exp(y_relboxes[1, anchor_i, cell_y, cell_x]) * anchor_box_sizes[anchor_i, 1]


    lower_y = cen_y - h * 0.5
    lower_x = cen_x - w * 0.5
    upper_y = cen_y + h * 0.5
    upper_x = cen_x + w * 0.5

    bboxes = np.concatenate([lower_y[:,None], lower_x[:,None], upper_y[:,None], upper_x[:,None]], axis=1)
    if y_obj_confidence is None:
        box_conf = np.ones((bboxes.shape[0],))
    elif isinstance(y_obj_confidence, float):
        box_conf = np.ones((bboxes.shape[0],)) * y_obj_confidence
    elif isinstance(y_obj_confidence, np.ndarray):
        if y_obj_class is None:
            box_conf = y_obj_confidence.flatten()
        else:
            box_conf = y_obj_confidence[anchor_i, cell_y, cell_x]
    else:
        raise TypeError('y_obj_confidence should be None, a float or a NumPy array, not a {}'.format(
            type(y_obj_confidence)))

    if y_obj_class is None:
        box_classes = None
    else:
        box_classes = y_obj_class[anchor_i, cell_y, cell_x]

    return bboxes, box_classes, box_conf


def clip_bboxes_to_image_shape(bboxes, image_shape):
    img_h, img_w = image_shape
    lower_y = np.clip(bboxes[:,0:1], 0.0, float(img_h))
    lower_x = np.clip(bboxes[:,1:2], 0.0, float(img_w))
    upper_y = np.clip(bboxes[:,2:3], 0.0, float(img_h))
    upper_x = np.clip(bboxes[:,3:4], 0.0, float(img_w))
    return np.concatenate([lower_y, lower_x, upper_y, upper_x], axis=1)


def non_maximal_suppression(bboxes, box_conf, iou_threshold):
    """
    Implementation heavily inspired by https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    order = np.argsort(-box_conf)
    n = box_conf.shape[0]

    y0 = bboxes[:, 0]
    x0 = bboxes[:, 1]
    y1 = bboxes[:, 2]
    x1 = bboxes[:, 3]
    area = (y1 - y0) * (x1 - x0)

    i = 0
    valid = []
    while order.shape[0] > 0:
        box_i = order[0]
        valid.append(box_i)

        inter_y0 = np.maximum(y0[box_i], y0[order[1:]])
        inter_x0 = np.maximum(x0[box_i], x0[order[1:]])
        inter_y1 = np.minimum(y1[box_i], y1[order[1:]])
        inter_x1 = np.minimum(x1[box_i], x1[order[1:]])

        inter_h = np.maximum(inter_y1 - inter_y0, 0.0)
        inter_w = np.maximum(inter_x1 - inter_x0, 0.0)
        inter_area = inter_h * inter_w
        iou =  inter_area / (area[box_i] + area[order[1:]] - inter_area)

        not_suppressed_indices = np.where(iou <= iou_threshold)[0]
        order = order[not_suppressed_indices + 1]

    if len(valid) > 0:
        return np.array(valid)
    else:
        return np.zeros((0,), dtype=int)


def evaluate_predictions(gt_bboxes, pred_bboxes, pred_box_conf, iou_threshold):
    order = np.argsort(-pred_box_conf)
    pred_bboxes = pred_bboxes[order]
    pred_box_conf = pred_box_conf[order]

    n_gt = gt_bboxes.shape[0]
    gt_detected = np.zeros((gt_bboxes.shape[0],), dtype=bool)
    true_pos = np.zeros((order.shape[0],), dtype=int)
    false_pos = np.zeros((order.shape[0],), dtype=int)

    gt_h = np.maximum(gt_bboxes[:,2] - gt_bboxes[:,0], 0.0)
    gt_w = np.maximum(gt_bboxes[:,3] - gt_bboxes[:,1], 0.0)
    gt_area = gt_h * gt_w

    pred_h = np.maximum(pred_bboxes[:,2] - pred_bboxes[:,0], 0.0)
    pred_w = np.maximum(pred_bboxes[:,3] - pred_bboxes[:,1], 0.0)
    pred_area = pred_h * pred_w

    for pred_i in range(order.shape[0]):
        pred_bb = pred_bboxes[pred_i, :]

        inter_y0 = np.maximum(pred_bb[0], gt_bboxes[:, 0])
        inter_x0 = np.maximum(pred_bb[1], gt_bboxes[:, 1])
        inter_y1 = np.minimum(pred_bb[2], gt_bboxes[:, 2])
        inter_x1 = np.minimum(pred_bb[3], gt_bboxes[:, 3])

        inter_h = np.maximum(inter_y1 - inter_y0, 0.0)
        inter_w = np.maximum(inter_x1 - inter_x0, 0.0)
        inter_area = inter_h * inter_w
        iou = inter_area / (pred_area[pred_i] + gt_area - inter_area)

        max_iou = np.max(iou)
        max_iou_gt_ndx = np.argmax(iou)

        if max_iou > iou_threshold:
            if not gt_detected[max_iou_gt_ndx]:
                true_pos[pred_i] = 1
                gt_detected[max_iou_gt_ndx] = True
            else:
                false_pos[pred_i] = 1
        else:
            false_pos[pred_i] = 1

    true_pos = np.cumsum(true_pos)
    false_pos = np.cumsum(false_pos)

    return pred_box_conf, true_pos, false_pos, n_gt


import unittest

class TestCase_RCNNLocaliserData (unittest.TestCase):
    def test_convert_centre_size_box_bbox(self):
        bbox0 = np.array([3.5, 10.5, 4.5, 13.5])
        cs_box0 = np.array([4.0, 12.0, 1.0, 3.0])

        bbox1 = np.array([5.0, 14.0, 7.0, 16.0])
        cs_box1 = np.array([6.0, 15.0, 2.0, 2.0])

        bboxes = np.append(bbox0[None, :], bbox1[None, :], axis=0)
        cs_boxes = np.append(cs_box0[None, :], cs_box1[None, :], axis=0)

        self.assertTrue((centre_size_box_to_bbox(cs_box0) == bbox0).all())
        self.assertTrue((bbox_to_centre_size_box(bbox0) == cs_box0).all())

        self.assertTrue((centre_size_box_to_bbox(cs_boxes) == bboxes).all())
        self.assertTrue((bbox_to_centre_size_box(bboxes) == cs_boxes).all())


    def test_get_anchor_boxes_overlapping(self):
        self.assertTrue((get_anchor_boxes_overlapping(
            anchor_grid_origin=np.array([2,7]),
            anchor_grid_cell_size=np.array([4, 8]),
            anchor_grid_shape=(3,3),
            anchor_box_sizes=np.array([[2,2], [1,1]]),
            bbox=np.array([3.5,11.5,4.5,12.5])) == np.array([[[1,1],[1,1]], [[1,1], [1,1]]])).all())
        self.assertTrue((get_anchor_boxes_overlapping(
            anchor_grid_origin=np.array([2,7]),
            anchor_grid_cell_size=np.array([4, 8]),
            anchor_grid_shape=(3,3),
            anchor_box_sizes=np.array([[2,2], [1,1]]),
            bbox=np.array([5,14,7,16])) == np.array([[[1,2],[1,2]], [[1,2],[1,2]]])).all())
        self.assertTrue((get_anchor_boxes_overlapping(
            anchor_grid_origin=np.array([2,7]),
            anchor_grid_cell_size=np.array([4, 8]),
            anchor_grid_shape=(3,3),
            anchor_box_sizes=np.array([[2,2], [1,1]]),
            bbox=np.array([3,8,9,22])) == np.array([[[0,3],[0,3]], [[1,2],[1,2]]])).all())
        self.assertTrue((get_anchor_boxes_overlapping(
            anchor_grid_origin=np.array([2,7]),
            anchor_grid_cell_size=np.array([4, 8]),
            anchor_grid_shape=(3,3),
            anchor_box_sizes=np.array([[2,2], [1,1]]),
            bbox=np.array([2,7,10,23])) == np.array([[[0,3],[0,3]], [[0,3],[0,3]]])).all())


    def test_get_anchor_boxes_in_range(self):
        self.assertTrue((get_anchor_boxes_in_range(anchor_grid_origin=np.array([2,2]),
                                                   anchor_grid_cell_size=np.array([2,2]),
                                                   anchor_grid_shape=(20,20),
                                                   anchor_box_sizes=np.array([[8,16], [4,4], [1,1]]),
                                                   img_shape=np.array([42,42])) == np.array([[[1,19],[3,17]], [[0,20],[0,20]], [[0,20],[0,20]]])).all())


    def test_compute_coverage(self):
        cvg = compute_coverage(anchor_grid_origin=np.array([0,0]),
                               anchor_grid_cell_size=np.array([2,3]),
                               anchor_grid_shape=(9,9),
                               anchor_range=np.array([[0,9],[0,9]]),
                               anchor_box_size=np.array([4,6]),
                               bbox_lower=np.array([4,5]), bbox_upper=np.array([12,17]), bbox_area=96)

        for j in range(9):
            for i in range(9):
                self.assertEqual(cvg[j,i], _box_i_over_u((4, 12), (5, 17), (j * 2 - 2, j * 2 + 2), (i * 3 - 3, i * 3 + 3)))


    def test_ground_truth_boxes_to_y(self):
        y_obj, y_obj_mask, y_rel, y_box_mask = ground_truth_boxes_to_y(anchor_grid_origin=np.array([8,8]),
                                                                       anchor_grid_cell_size=np.array([16,16]),
                                                                       anchor_grid_shape=(30,40),
                                                                       anchor_box_sizes=np.array([[16,16], [8,8], [16,8], [8,16]]),
                                                                       img_shape=np.array([480,640]),
                                                                       ground_truth_bboxes=np.array([
                                                           [12.5,21,27.5,39], [196,294,204,306]
                                                       ]),
                                                                       coverage_lower_threshold=0.1,
                                                                       coverage_upper_threshold=0.2)

        # Compute coverage; tested above
        coverage = np.zeros((1,4,30,40))
        for i, anchor_box_size in enumerate([[16,16], [8,8], [16,8], [8,16]]):
            for j, bbox in enumerate([[12.5,21,27.5,39], [196,294,204,306]]):
                bbox = np.array(bbox)
                bbox_lower = bbox[:2]
                bbox_upper = bbox[2:]
                bbox_area = np.prod(bbox_upper - bbox_lower)
                cvg = compute_coverage(anchor_grid_origin=np.array([8,8]),
                                       anchor_grid_cell_size=np.array([16,16]),
                                       anchor_grid_shape=(30,40),
                                       anchor_range=np.array([[0,30],[0,40]]),
                                       anchor_box_size=np.array(anchor_box_size),
                                       bbox_lower=bbox_lower, bbox_upper=bbox_upper, bbox_area=bbox_area)
                coverage[0,i,:,:] = np.maximum(coverage[0,i,:,:], cvg)

        # y_obj should be == 1 where coverage > 0.2
        self.assertTrue((y_obj == (coverage > 0.2)).all())
        # y_mask, last dim=0 (learn objectness classifier) should be == 1 where coverage < 0.1 or coverage > 0.2
        self.assertTrue((y_obj_mask[:,:] == ((coverage > 0.2) | (coverage < 0.1))).all())
        # y_mask, last dim=1 (learn bbox regressor) should be == 1 where coverage > 0.2
        self.assertTrue((y_box_mask[:,:] == (coverage > 0.2)).all())

        # Should be 7 places where a box is being learned
        self.assertEqual(7, y_box_mask[:,:].sum())
        # Check them all
        self.assertEqual(1, y_box_mask[0,1,1])
        self.assertEqual(0, y_box_mask[1,1,1])
        self.assertEqual(1, y_box_mask[2,1,1])
        self.assertEqual(1, y_box_mask[3,1,1])
        self.assertEqual(1, y_box_mask[0,12,18])
        self.assertEqual(1, y_box_mask[1,12,18])
        self.assertEqual(1, y_box_mask[2,12,18])
        self.assertEqual(1, y_box_mask[3,12,18])

        # Check the boxea
        b0a = _rel_box(np.array([20,30,15,18]), np.array([24,24,16,16]))
        b0c = _rel_box(np.array([20,30,15,18]), np.array([24,24,16,8]))
        b0d = _rel_box(np.array([20,30,15,18]), np.array([24,24,8,16]))

        b1a = _rel_box(np.array([200,300,8,12]), np.array([200,296,16,16]))
        b1b = _rel_box(np.array([200,300,8,12]), np.array([200,296,8,8]))
        b1c = _rel_box(np.array([200,300,8,12]), np.array([200,296,16,8]))
        b1d = _rel_box(np.array([200,300,8,12]), np.array([200,296,8,16]))

        self.assertTrue(np.allclose(y_rel[:,0,1,1], b0a))
        self.assertTrue(np.allclose(y_rel[:,2,1,1], b0c))
        self.assertTrue(np.allclose(y_rel[:,3,1,1], b0d))
        self.assertTrue(np.allclose(y_rel[:,0,12,18], b1a))
        self.assertTrue(np.allclose(y_rel[:,1,12,18], b1b))
        self.assertTrue(np.allclose(y_rel[:,2,12,18], b1c))
        self.assertTrue(np.allclose(y_rel[:,3,12,18], b1d))


    def test_ground_truth_boxes_to_y_with_box_classes(self):
        y_obj1, y_obj_mask, y_rel, y_box_mask = ground_truth_boxes_to_y(anchor_grid_origin=np.array([8,8]),
                                                                        anchor_grid_cell_size=np.array([16,16]),
                                                                        anchor_grid_shape=(30,40),
                                                                        anchor_box_sizes=np.array([[16,16], [8,8], [16,8], [8,16]]),
                                                                        img_shape=np.array([480,640]),
                                                                        ground_truth_bboxes=np.array([
                                                           [12.5, 21, 27.5, 39]
                                                       ]),
                                                                        coverage_lower_threshold=0.1,
                                                                        coverage_upper_threshold=0.2)

        y_obj2, y_obj_mask, y_rel, y_box_mask = ground_truth_boxes_to_y(anchor_grid_origin=np.array([8,8]),
                                                                        anchor_grid_cell_size=np.array([16,16]),
                                                                        anchor_grid_shape=(30,40),
                                                                        anchor_box_sizes=np.array([[16,16], [8,8], [16,8], [8,16]]),
                                                                        img_shape=np.array([480,640]),
                                                                        ground_truth_bboxes=np.array([
                                                           [196, 294, 204, 306]
                                                       ]),
                                                                        coverage_lower_threshold=0.1,
                                                                        coverage_upper_threshold=0.2)

        y_obj, y_obj_mask, y_rel, y_box_mask = ground_truth_boxes_to_y(anchor_grid_origin=np.array([8,8]),
                                                                       anchor_grid_cell_size=np.array([16,16]),
                                                                       anchor_grid_shape=(30,40),
                                                                       anchor_box_sizes=np.array([[16,16], [8,8], [16,8], [8,16]]),
                                                                       img_shape=np.array([480,640]),
                                                                       ground_truth_bboxes=np.array([
                                                           [12.5, 21, 27.5, 39], [196, 294, 204, 306]
                                                       ]),
                                                                       box_classes=np.array([1, 2]),
                                                                       coverage_lower_threshold=0.1,
                                                                       coverage_upper_threshold=0.2)

        y_obj_expected = np.zeros_like(y_obj1)
        y_obj_expected[y_obj1 > 0] = 1
        y_obj_expected[y_obj2 > 0] = 2

        # y_obj should be == 1 where coverage > 0.2
        self.assertTrue((y_obj == y_obj_expected).all())



    def test_ground_truth_boxes_to_y_outside_screen_bounds(self):
        y_obj, y_obj_mask, y_rel, y_box_mask = ground_truth_boxes_to_y(anchor_grid_origin=np.array([8,8]),
                                                                       anchor_grid_cell_size=np.array([16,16]),
                                                                       anchor_grid_shape=(30,40),
                                                                       anchor_box_sizes=np.array([[32,32], [16,16], [32,16], [16,32]]),
                                                                       img_shape=np.array([480,640]),
                                                                       ground_truth_bboxes=np.array([
                                                           [12.5, 21, 27.5, 39], [196, 294, 204, 306]
                                                       ]),
                                                                       coverage_lower_threshold=0.0001,
                                                                       coverage_upper_threshold=0.0001)

        # There should be a border of 0's around the edge as some of the boxes go outside the bounds of the screen
        self.assertFalse((y_obj_mask[:,:,:] == 1).all())
        # But the inner part of the mask should be 1
        self.assertTrue((y_obj_mask[:,1:-1,1:-1] == 1).all())

        # The border should be 0 all over for anchor box 0 (32x32)
        self.assertTrue((y_obj_mask[0,:,:1] == 0).all())
        self.assertTrue((y_obj_mask[0,:,-1:] == 0).all())
        self.assertTrue((y_obj_mask[0,:1,:] == 0).all())
        self.assertTrue((y_obj_mask[0,-1:,:] == 0).all())
        # The border should be 1 all over for anchor box 1 (16x16)
        self.assertTrue((y_obj_mask[1,:,:1] == 1).all())
        self.assertTrue((y_obj_mask[1,:,-1:] == 1).all())
        self.assertTrue((y_obj_mask[1,:1,:] == 1).all())
        self.assertTrue((y_obj_mask[1,-1:,:] == 1).all())
        # The border should be 1 on the left and right edges, 0 on top and bottom for anchor box 2 (16x32)
        self.assertTrue((y_obj_mask[2,1:-1,:1] == 1).all())
        self.assertTrue((y_obj_mask[2,1:-1,-1:] == 1).all())
        self.assertTrue((y_obj_mask[2,:1,:] == 0).all())
        self.assertTrue((y_obj_mask[2,-1:,:] == 0).all())
        # The border should be 1 on the top and bottom edges, 0 on left and right for anchor box 3 (16x32)
        self.assertTrue((y_obj_mask[3,:1,1:-1] == 1).all())
        self.assertTrue((y_obj_mask[3,-1:,1:-1] == 1).all())
        self.assertTrue((y_obj_mask[3,:,:1] == 0).all())
        self.assertTrue((y_obj_mask[3,:,-1:] == 0).all())
