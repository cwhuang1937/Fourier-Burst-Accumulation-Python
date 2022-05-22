from utils import *
from SuperGlue import *
import argparse

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help='The path of the burst.')
parser.add_argument(
    "-e", help='The file extension of the input images.', choices=['jpg', 'png'], type=str, default='jpg')
parser.add_argument(
    "-f", '--first', help='Use the first image as the alignment reference image.', action="store_true")
parser.add_argument(
    "-k", '--ksize', help='The value of gaussian kernel size, default=31.', type=int, default=31)
parser.add_argument(
    "-ks", help='The value of ks, default=50.', type=int, default=50)
parser.add_argument(
    "-n", help='Use the paper\'s method to do the post-processing.', action="store_true")
parser.add_argument(
    "-p", help='The value of p, default=11.', type=int, default=11)
parser.add_argument(
    "-s", '--sift', help='Use SIFT.', action="store_true")

args = parser.parse_args()
ks = args.ks
Use_SIFT = args.sift
select_ref_img = not args.first
burst_path = args.path
file_extension = f'*.{args.e}'
gaussian_ksize = args.ksize  # gaussian kernel size
p = args.p
paper_post = args.n
print('Arguments:')
print(f'file extension: {file_extension}')
print(f'path: {burst_path}')
print(
    f'Use the first image as the alignment reference image: {not select_ref_img}')
print(f'gaussian kernel size: {gaussian_ksize}')
print(f'ks: {ks}')
print(f'Use the paper\'s method to do the post-processing.: {paper_post}')
print(f'p: {p}')
print(f'path: {burst_path}')
print(f'use SIFT: {Use_SIFT}')
print('-'*50)

# Read images
burst = read_burst(burst_path, file_extension)
if len(burst > 0):
    print(f'Successfully read {len(burst)} images.')
else:
    raise RuntimeError('Failed to read images.')
print('-'*50)

# select reference image
if select_ref_img:
    print('Selecting the clearest image as the reference image...')
    change_reference(burst)
    print('-'*50)

# Bursr Registration
# reference source: `https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/`
print('Aligning images...')
if Use_SIFT:
    burst = register_burst(burst)
else:
    burst = register_burst_SuperGlue(burst, copy=False, force_cpu=False)
# cut the edge in order to remove black pixels
burst = burst[:, 20:-20, 20:-20, :]
print('-'*50)

# FBA
print('Aggregating images...')
# np.moveaxis: change the shape from (# of img, R, C, color) to (# of img, color, R, C)
spectrums = np.fft.fft2(np.moveaxis(burst, 3, 1))
# spectrum.shape = (# of img, color, R, C)

# get the spectrum of a blur kernel
shape = spectrums.shape[-2:]
sig = min(shape)/ks
# blur_kernel_spectrum=get_gau_ker(gaussian_ksize, sig, shape)[1]

# average color channels
weight = np.mean(np.abs(spectrums), axis=1)

# pass through the gaussian filter
weight = np.fft.fftshift(weight)
for i in range(weight.shape[0]):
    weight[i] = cv2.GaussianBlur(weight[i, :, :], (31, 31), sig)
weight = np.fft.ifftshift(weight)

weight = np.power(weight, p)
weight /= np.sum(weight, axis=0)

# expand the shape of the weight from (# of img, R, C) to (# of img, color, R, C)
weight = np.repeat(np.expand_dims(weight, axis=1), 3, axis=1)

# restore image
spectrum_restored = np.sum(weight*spectrums, axis=0)
image_restored = np.fft.ifft2(spectrum_restored)

# change the shape from (color, R, C) to (R, C, color)
image_restored = np.moveaxis(image_restored, 0, 2)

# restore to uint8
image_restored = image_restored.real
image_restored = np.where(image_restored < 0, 0, image_restored)
image_restored = np.where(image_restored > 255, 255,
                          image_restored).astype(np.uint8)
print('-'*50)

# Post processing
print('Post processing...')

if paper_post:
    result = paper_post_processing(image_restored, gaussian_ksize)
else:
    result = our_post_processing(image_restored, gaussian_ksize)

cv2.imwrite(f'result.{args.e}', image_restored)
cv2.imwrite(f'result_post_processing.{args.e}', result)
print('-'*50)
print('Done')
