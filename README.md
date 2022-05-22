# Fourier Burst Accumulation
## How to use
```
usage: python3 FBA.py [-h] [-e {jpg,png}] [-f] [-k KSIZE] [-ks KS] [-n] [-p P] [-s] path

positional arguments:
  path                  The path of the burst.

optional arguments:
  -h, --help            show this help message and exit
  -e {jpg,png}          The file extension of the input images.
  -f, --first           Use the first image as the alignment reference image.
  -k KSIZE, --ksize KSIZE
                        The value of gaussian kernel size, default=31.
  -ks KS                The value of ks, default=50.
  -n                    Use the paper's method to do the post-processing.
  -p P                  The value of p, default=11.
  -s, --sift            Use SIFT.
```
### Example
```
python3 FBA.py images/dataset/anthropologie
```
## Output
```
result: The result without post-processing
result_post_processing: The post-processing result
```
## images
### dataset
1. The images in the dataset are provided by the author of the paper. [link](http://dev.ipol.im/~mdelbra/fba/)
2. The images in the test folder in the dataset are a burst of images that we rearrange their order. We put a blurred image as the first image.
3. The results offered by the author are in `images/dataset/[data set name]/author_result`.
4. The results processed by our code are in `images/our_code_result`.
   ```
   images/our_code_result/our_post_processing: The results are processed by our code with our post-processing method.
   images/our_code_result/paper_post_processing: The results are processed by our code with the paper's post-processing method.
   ```
## References
1. [M. Delbracio and G. Sapiro, "Removing Camera Shake via Weighted Fourier Burst Accumulation," in IEEE Transactions on Image Processing, vol. 24, no. 11, pp. 3293-3307, Nov. 2015, doi: 10.1109/TIP.2015.2442914.](https://ieeexplore.ieee.org/document/7120097)
2. [Delbracio, Mauricio, and Guillermo Sapiro. "Burst deblurring: Removing camera shake through fourier burst accumulation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.](https://ieeexplore.ieee.org/document/7298852)
3. [Paul-Edouard Sarlin. “SuperGlue: Learning Feature Matching with Graph Neural Networks.” CVPR 2020](https://ieeexplore.ieee.org/document/9157489)
4. [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
5. [MATLAB code](https://github.com/remicongee/Fourier-Burst-Accumulation)
