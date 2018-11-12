# Assignment 4
# CS 152
# Fall, 2018
# non_max_suppression.py

# Richy, Jason

import numpy as np

def nms(boxes, scores, classes, overlap=0.45, min_score=.4):
    """ Does Non Max Suppression on the given boxes, scores, and classes.
    Any boxes with score < min_score are ignored. Boxes are chosen based on
    decreasing score.
    When a box is chosen, any overlapping boxes of the same class with IoU < overlap
    are discarded.

    Parameters:
      boxes: Numpy array of shape (k, 4). Each row consists of left, top, right, and bottom.
      scores: NumPy array of shape (k). Each value is in the range [0.0, 1.0]
      classes: NumPy array of shape (k). Each value is an integer in the range 0..100.


    Return result:
      a NumPy array with j elements in it, each representing a chosen box.
      Each of the j elements is an index into the original array.
    """

    # make a numpy array of 0 zeros
    prediction = np.zeros(0)

    # sort the indices and store the highest score index
    indices = np.argsort(scores)[::-1]

    # only take in the ones greater than our min score, otherwise we don't need it
    while indices.size > 0 and scores[indices[0]] > min_score:
      #  store the first index
      max_i = indices[0]
      # delete the first index in order to go through the indices
      indices = np.delete(indices, 0)
      # add the max index to the prediction array
      prediction = np.append(prediction, max_i)
      # make an array of boolean values that checks whether the class of max_i matches
      # class of all the indices
      same_classes = np.equal(classes[indices], classes[max_i])
      # check if iou is greater than the overlap value of 0.45
      overlaps = np.greater_equal(iou(boxes[indices], boxes[max_i]), overlap)
      #
      to_discard = np.logical_and(same_classes, overlaps)
      # reverse the boolean values
      indices = indices[np.logical_not(to_discard)]
    return prediction

def iou(boxes, box):
  # essentially get the min of boxes
  topleft = (np.clip(boxes, box, None))
  # essentially get the max of the boxes
  bottomright = (np.clip(boxes, None, box))

  # trim the boxes to get the intersecting box of the topleft and bottom right corners
  topleft = topleft[:,:2]
  bottomright = bottomright[:,2:]

  together = bottomright - topleft

  # calculate the overlap values
  overlap = together * together[:,-1:]
  overlap = overlap[:,:1].T[0]

  # calculations: area of boxes and the max area and the union
  areas = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
  max_i_area = (box[3]-box[1])*(box[2]-box[0])
  union = areas - overlap + max_i_area
   # compute the overlap of ratio between smaller region and bounding box
  return np.divide(overlap, union, where=union!=0)
