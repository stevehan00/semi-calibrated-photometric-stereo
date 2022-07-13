# Semi-Calibrated Photometric Stereo Implementation

This repository includes implementation of "Semi-Calibrated Photometric Stereo, TPAMI 2018". Photometric stereo is a technique estimating surface normal of an object under varying light conditions. "Semi-Calibrated" means light intensities are unknown, but the light directions are known. 



## Overview

Factorization-based and Alternating Minimization method are implemented. Methods are tested using two objects in DiLiGent dataset. 


## Dependencies

```
python == 3.8.5
numpy == 1.19.2
matplotlib == 3.3.2
scikit-learn == 0.23.2
```



## Examples

Under normal maps are derived from an alternating minimization method. Mean angular error(MAE) of BEAR is about 9.5 and CAT has about 9.0. Then I showed an error map about each object.

<img src="img\bear_normal_AM.PNG" alt="bear AM" style="zoom:50%;" />

<img src="img\cat_normal_AM.PNG" alt="cat AM" style="zoom:50%;" />



## Testing

```
python test.py
```



## Dataset
https://sites.google.com/site/photometricstereodata/
