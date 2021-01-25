# Semi-Calibrated Photometric Stereo Implementation

This repository includes implementation of paper "Semi-Calibrated Photometric Stereo, TPAMI 2020". Photometric stereo is a technique estimating surface normal of an object through observation under varying light conditions. "Semi-Calibrated" means light intensities are unknown, but the light directions are known. 



## Overview

Factorization-based and Alternating Minimization method is implemented except for others. Methods are tested using parts of DiLiGent dataset. And I include data in this repository. Factorization-based method is developed because Linear-joint estimation is limited at practical setting. That is related with SVD's Time Complexity. Then, this paper showed an Alternating minimization method avoiding the round-off error. In results, AM method has the higher precision compared to other methods.



## Dependencies

```
python : 3.8.5
numpy : 1.19.2
matplotlib : 3.3.2
scikit-learn : 0.23.2
```



## Examples

Under normal maps are derived from an Alternating minimization method. Mean Angular Error of BEAR is about 9.5 and CAT has about 9.0. Then I showed an error map about each object.

<img src="img\bear_normal_AM.PNG" alt="bear AM" style="zoom:50%;" />

<img src="img\cat_normal_AM.PNG" alt="cat AM" style="zoom:50%;" />



## Testing

```
python test.py
```



## dataset
https://sites.google.com/site/photometricstereodata/
