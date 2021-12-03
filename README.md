# Low Light Image Enhancement
Course Project for the course Image Processing (EE610) offered in Autumn 2021, IIT Bombay. 

## Methods Followed
- Retinex Theory and Dual-Tree Complex Wavelet Transform approach based on [[1]](#1)
- Retinex Theory approach based on estimating illumination map of image (LIME) [[2]](#2)
- Patch-wise pixel predicting Deep Learning Model 

## Results
Best results are obtained using the LIME method 
| ![Original](/images/111.png) | ![Lime Result](/images/111_lime.png) |
|:--:| :--:|
| *Original Low Light Image* | *LIME enhanced image*|

## References
<a id="1">[1]</a> 
M. xiang Yang, G. jin Tang, X. hua Liu, L. qian Wang, Z. guan Cui, and S. huai Luo, “Low-light image enhancement based on retinex theory and dual-tree complex wavelet transform,” Optoelectronics Letters, vol. 14, no. 6, pp. 470–475, 2018.

<a id="2">[2]</a> 
X. Guo, Y. Li, and H. Ling, “Lime: Low-light image enhancement via illumination map estimation,” IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982–993, 2017.