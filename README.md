# Non-MaxSuppression

I implemented a Non-Max Suppression function in order to help detect and label various objects (computer vision). I also implemented a unittest class in order to define unit tests and check whether the function performed as expected. 

Moreover, this project was purposely implemented by only using one overall loop and avoiding any unnecessary for loop calls. Instead, the additional loops were replaced with vectorized code in order to obtain a substantially faster speed.  

The Non-Max Suppression algorithm was implemented as follows:
1. Discard all boxes with the probability of an object (Pc) < 0.4
2. While boxes remain:
    1. Choose box with highest Pc and  mark that as a prediction
    2. Discard all boxes which predict the same class and overlap that
       prediction with IoU > 0.45

Notes: 
* IOU = intersection over union.
* Project was completed with the help of NumPy
*  [Numpy index arrays](https://docs.scipy.org/doc/numpy-1.15.1/user/basics.indexing.html#index-arrays)

