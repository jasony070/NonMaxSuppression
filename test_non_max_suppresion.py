# Assignment 4
# CS 152
# Fall, 2018
# test_non_max_suppression.py

# Richy, Jason

from collections import Counter
from datetime import date
from non_max_suppression import nms
from non_max_suppression import iou
import numpy as np
import random
import unittest




class TestNonMaxSuppression(unittest.TestCase):

    def assertEither(self,x,y):
      """ check whether it is in either order """
      self.assertTrue(len(x) == 1 and x in y)

    # Student tests
    def test_different_classes_lowscore(self):
      """ the scores are less than 0.4 so it should not return anything """
      boxes = np.asarray([(1,2,3,4),(2,3,5,6)])
      classes = np.asarray([1,2])
      scores = np.asarray([0.2,0.3])
      self.assertEqualAnyOrder(nms(boxes, scores, classes), np.asarray([]))

    def test_same_classes_lowscore(self):
      """ the scores are less than 0.4 but the classes are the same, so it
          should not return anything """
      boxes = np.asarray([(1,2,3,4),(2,3,5,6)])
      classes = np.asarray([1,1])
      scores = np.asarray([0.2,0.3])
      self.assertEqualAnyOrder(nms(boxes, scores, classes), np.array([]))

    def test_low_iou_different_classes(self):
      """ the iou is greater than 0.45 but the boxes are different classes, so
          it should return the indices of all the boxes """
      boxes = np.asarray([[1,2,3,4],[2,3,5,6]])
      classes = np.asarray([1,2])
      scores = np.asarray([0.5,0.55])
      self.assertEqualAnyOrder(nms(boxes, scores, classes), np.asarray([0,1]))

    def test_high_iou_same_class(self):
      """ the boxes are the same classes but the iou is high so you should
          the box index with the highest score """
      boxes = np.asarray([(1,1,2,2),(1.01,1,2.01,2)])
      classes = np.asarray([1,1])
      scores = np.asarray([0.6,0.5])
      self.assertEqualAnyOrder(nms(boxes, scores, classes), np.array([0]))

    def test_high_iou_different_class(self):
      """ the boxes are different classes and the iou is high but since
          they are different classes, it should return all the boxes """
      boxes = np.asarray([(1,1,2,2),(1.01,1,2.01,2)])
      classes = np.asarray([1,2])
      scores = [0.6,0.5]
      self.assertEqualAnyOrder(nms(boxes, scores, classes), np.array([0,1]))


    def test_same_boxes_same_classes_samescores(self):
      """ everything is the same so it should return either of the boxes """
      boxes = np.asarray([(1,2,3,4),(1,2,3,4)])
      classes = np.asarray([1,1])
      scores = np.asarray([0.5,0.5])
      self.assertEither(nms(boxes, scores, classes), np.array([(1,2,3,4)]))


    def assertEqualAnyOrder(self, a, b):
        """ Asserts that a and b contain the same values, in any order.
        Assumes that a and b are one-dimensional NumPy arrays. """
        a.sort()
        b.sort()
        self.assertEqual(list(a), list(b))


    def test_no_boxes_returns_empty(self):
        empty = np.array([])
        self.assertEqual(0, nms(empty, empty, empty).size)


    def test_random_overlapping(self):
        """Tests by creating a random set of boxes: one true one per class
        and a number of random ones.

        Don't start using this test until you've got all other,
        more basic tests working."""

        def random_in_range(a_min, a_max):
            return a_min + random.random() * (a_max - a_min)

        # Use today's date as a seed, so that we are reproducible all day
        # today (while debugging:)
        random.seed(str(date.today()))
        MIN_COORD = 0
        MAX_COORD = 1000

        for run in range(20):
            boxes = []
        MAX_COORD = 1000

        for run in range(20):
            boxes = []
            scores = []
            classes = []
            good_indexes = []
            for c in range(99):
                # For each class, create one real box, and a number
                # of boxes with lower scores that overlap the real box
                left = random.randrange(MAX_COORD-1)
                top = random.randrange(MAX_COORD-1)
                right = random.randrange(left+1, MAX_COORD)
                bottom = random.randrange(top+1, MAX_COORD)
            scores = []
            classes = []
            good_indexes = []
            for c in range(99):
                # For each class, create one real box, and a number
                # of boxes with lower scores that overlap the real box
                left = random.randrange(MAX_COORD-1)
                top = random.randrange(MAX_COORD-1)
                right = random.randrange(left+1, MAX_COORD)
                bottom = random.randrange(top+1, MAX_COORD)
                real_box = [left, top, right, bottom]
                real_score = random_in_range(.3, .9)
                boxes.append(real_box)
                scores.append(real_score)
                classes.append(c)
                if real_score > 0.4:
                    good_indexes.append(len(boxes) - 1)
                for fake_boxes in range(20):
                    h_factor = random_in_range(-.2, .2)
                    v_factor = random_in_range(-.2, .2)
                    delta_h = (right - left) * h_factor
                    delta_v = (bottom - top) * v_factor
                    fake_box = [
                            left + delta_h, top + delta_v,
                            right + delta_h, bottom + delta_v]
                    boxes.append(fake_box)
                    fake_score = random_in_range(real_score - .2, real_score - .001)
                    scores.append(fake_score)
                    classes.append(c)

                self.assertEqualAnyOrder(np.array(good_indexes),
                    nms(np.array(boxes), np.array(scores), np.array(classes)))

if __name__ == '__main__':
    unittest.main()

