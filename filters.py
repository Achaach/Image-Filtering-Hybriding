# PyTorch tutorial on constructing neural networks:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import os
from typing import List, Tuple, Union
import numpy as np
import math 
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision


def create_1d_gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:

    kernel = torch.FloatTensor()


    standard_deviation = int(standard_deviation)
    k = 4 * standard_deviation + 1
    mean = int(k / 2)
    Z = math.sqrt(2 * math.pi * standard_deviation**2)
    x = torch.linspace(0, k-1, int(k))
    kernel = 1/Z * torch.exp(-(x - mean)**2/(2*standard_deviation**2))
    sum = torch.sum(kernel)
    kernel = kernel / sum


    return kernel

def my_1d_filter(signal: torch.FloatTensor,
                 kernel: torch.FloatTensor) -> torch.FloatTensor:

    filtered_signal = torch.FloatTensor()
    signal_list = signal.tolist()
    kernel_list = kernel.tolist()
    signal_len = len(signal)
    kernel_len = len(kernel)
    
    i = 0
    output = []
    if kernel_len % 2 == 1:
        if (kernel_len - 1) / 2 == 1:
            zero_pad = [0]
        else:
            pad = (kernel_len - 1) / 2
            zero_pad = [0] * int(pad)    
    else:
        pad = kernel_len / 2
        zero_pad = [0] * int(pad) 

    new_signal_list = zero_pad + signal_list + zero_pad    

    while i < signal_len:
        total = 0
        sub_signal = new_signal_list[i: i+kernel_len]
        for num1, num2 in zip(sub_signal, kernel_list):
            current = num1 * num2
            total = current + total
        i = i+1
        output.append(total)
    
    filtered_signal = torch.FloatTensor(output)

    return filtered_signal


# TODO - 3
def create_2d_gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    # Create a 2D Gaussian kernel using the specified standard deviation in
    # each dimension, and no cross-correlation between dimensions,
    #
    # i.e.
    # sigma_matrix = [standard_deviation^2    0
    #                 0                       standard_deviation^2]
    #
    # The kernel should have:
    # - shape (k, k) where k = standard_deviation * 4 + 1
    # - mean = floor(k / 2)
    # - values that sum to 1
    #
    # Args:
    #     standard_deviation (float): the standard deviation along a dimension
    # Returns:
    #     torch.FloatTensor: 2D Gaussian kernel
    #
    # HINT:
    # - The 2D Gaussian kernel here can be calculated as the outer product of two
    #   vectors drawn from 1D Gaussian distributions.

    kernel_2d = torch.Tensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    #k = int(4 * standard_deviation + 1)
    #mean = np.floor(k / 2)
    #var = standard_deviation ** 2

    #k_list = torch.arange(k)
    #kernel_1d = torch.exp(-1 / (2 * var) * ((k_list - mean) ** 2))
    #kernel_1d = kernel_1d / (torch.sum(kernel_1d))

    #kernel_2d = torch.ger(kernel_1d, kernel_1d)

    standard_deviation = int(standard_deviation)
    k = 4 * standard_deviation + 1
    mean = k // 2
    kernel_2d = torch.arange(-mean, mean + 1).reshape((-1, 1)) ** 2
    kernel_2d = kernel_2d + kernel_2d.transpose(1, 0)
    kernel_2d = torch.exp(-0.5 * (kernel_2d - mean) / standard_deviation ** 2)
    sum = torch.sum(kernel_2d)
    kernel_2d /= sum

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return kernel_2d


# TODO - 4
def my_imfilter(image, image_filter, image_name="Image"):
    # Apply a filter to an image. Return the filtered image.
    #
    # Args:
    #     image: Torch tensor of shape (m, n, c)
    #     filter: Torch tensor of shape (k, j)
    # Returns:
    #     filtered_image: Torch tensor of shape (m, n, c)
    #
    # HINTS:
    # - You may not use any libraries that do the work for you. Using torch to work
    #  with matrices is fine and encouraged. Using OpenCV or similar to do the
    #  filtering for you is not allowed.
    # - I encourage you to try implementing this naively first, just be aware that
    #  it may take a long time to run. You will need to get a function
    #  that takes a reasonable amount of time to run so that the TAs can verify
    #  your code works.
    # - Useful functions: torch.nn.functional.pad

    filtered_image = torch.Tensor()

    assert image_filter.shape[0] % 2 == 1
    assert image_filter.shape[1] % 2 == 1

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    m, n, c = image.size()

    k, j = image_filter.size()
    
    k_pad = int(np.floor(k / 2))
    j_pad = int(np.floor(j / 2))
    
    new_img = torch.nn.functional.pad(image, (0, 0, j_pad, j_pad, k_pad, k_pad))
    filtered_image = torch.zeros(m, n, c)
    
    for C in range(c):
        for M in range(m):
            for N in range(n):
                temp = new_img[M:M+k, N:N+j, C]
                filtered_image[M, N, C] = torch.sum(temp * image_filter)


    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return filtered_image


# TODO - 5
def create_hybrid_image(image1, image2, filter):
    # Take two images and a low-pass filter and create a hybrid image. Return
    # the low frequency content of image1, the high frequency content of image2,
    # and the hybrid image.
    #
    # Args:
    #     image1: Torch tensor of dim (m, n, c)
    #     image2: Torch tensor of dim (m, n, c)
    #     filter: Torch tensor of dim (x, y)
    # Returns:
    #     low_freq_image: Torch tensor of shape (m, n, c)
    #     high_freq_image: Torch tensor of shape (m, n, c)
    #     hybrid_image: Torch tensor of shape (m, n, c)
    #
    # HINTS:
    # - You will use your my_imfilter function in this function.
    # - You can get just the high frequency content of an image by removing its low
    #   frequency content. Think about how to do this in mathematical terms.
    # - Don't forget to make sure the pixel values of the hybrid image are between
    #   0 and 1. This is known as 'clipping' ('clamping' in torch).
    # - If you want to use images with different dimensions, you should resize them
    #   in the notebook code.

    hybrid_image = torch.Tensor()
    low_freq_image = torch.Tensor()
    high_freq_image = torch.Tensor()

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    low_freq_image = my_imfilter(image1, filter)
    low_freq_image2 = my_imfilter(image2, filter)

    high_freq_image = image2 - low_freq_image2

    hybrid_image = torch.clamp( (low_freq_image + high_freq_image), 0, 1)


    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return low_freq_image, high_freq_image, hybrid_image


# TODO - 6.1
def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    # Create a dataset of paired images from a directory.
    #
    # The dataset should be partitioned into two sets: one contains images that
    # will have the low pass filter applied, and the other contains images that
    # will have the high pass filter applied.
    #
    # Args:
    #     path: string specifying the directory containing images
    # Returns:
    #     images_a: list of strings specifying the paths to the images in set A,
    #         in lexicographically-sorted order
    #     images_b: list of strings specifying the paths to the images in set B,
    #         in lexicographically-sorted order

    images_a = []
    images_b = []

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    file_list = os.listdir(path)
    for fname in file_list:
        if 'a_' in fname:
            images_a.append(path + fname)
        elif 'b_' in fname:
            images_b.append(path + fname)

    images_a.sort()
    images_b.sort()
    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return images_a, images_b


# TODO - 6.2
def get_cutoff_standardddeviations(path: str) -> List[int]:
    # Get the cutoff standard deviations corresponding to each pair of images
    # from the cutoff_standarddeviations.txt file
    #
    # Args:
    #     path: string specifying the path to the .txt file with cutoff standard
    #         deviation values
    # Returns:
    #     List[int]. The array should have the same
    #         length as the number of image pairs in the dataset

    cutoffs = []

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    with open(path, "r") as f:
      cutoffs = f.read().split('\n')
    for i in cutoffs[::-1]:
      if i == '':
        cutoffs.pop(-1)
      else:
        break
    cutoffs = list(map(int, cutoffs))

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return cutoffs

# TODO - 6.3


class HybridImageDataset(data.Dataset):
    # Hybrid images dataset
    def __init__(self, image_dir: str, cf_file: str) -> None:
        # HybridImageDataset class constructor.
        #
        # You must replace self.transform with the appropriate transform from
        # torchvision.transforms that converts a PIL image to a torch Tensor. You can
        # specify additional transforms (e.g. image resizing) if you want to, but
        # it's not necessary for the images we provide you since each pair has the
        # same dimensions.
        #
        # Args:
        #     image_dir: string specifying the directory containing images
        #     cf_file: string specifying the path to the .txt file with cutoff
        #         standard deviation values

        images_a, images_b = make_dataset(image_dir)

        self.cutoffs = get_cutoff_standardddeviations(cf_file)

        self.transform = torchvision.transforms.ToTensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        self.images_a = images_a
        self.images_b = images_b

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

    def __len__(self) -> int:
        # Return the number of pairs of images in dataset

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

 
        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return len(self.images_a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Return the pair of images and corresponding cutoff standard deviation
        # value at index `idx`.
        #
        # Since self.images_a and self.images_b contain paths to the images, you
        # should read the images here and normalize the pixels to be between 0 and 1.
        # Make sure you transpose the dimensions so that image_a and image_b are of
        # shape (c, m, n) instead of the typical (m, n, c), and convert them to
        # torch Tensors.
        #
        # If you want to use a pair of images that have different dimensions from
        # one another, you should resize them to match in this function using
        # torchvision.transforms.
        #
        # Args:
        #     idx: int specifying the index at which data should be retrieved
        # Returns:
        #     image_a: Tensor of shape (c, m, n)
        #     image_b: Tensor of shape (c, m, n)
        #     cutoff: int specifying the cutoff standard deviation corresponding to
        #         (image_a, image_b) pair
        #
        # HINTS:
        # - You should use the PIL library to read images
        # - You will use self.transform to convert the PIL image to a torch Tensor

        image_a = torch.Tensor()
        image_b = torch.Tensor()
        cutoff = 0

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        image_a = PIL.Image.open(self.images_a[idx] )
        image_a = self.transform(image_a)
        image_b = PIL.Image.open(self.images_b[idx] )
        image_b = self.transform(image_b)
        cutoff = self.cutoffs[idx]

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return image_a, image_b, cutoff


# TODO - 7
class HybridImageModel(nn.Module):
    def __init__(self):
        # Initializes an instance of the HybridImageModel class.
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_standarddeviation: int) -> torch.Tensor:
        # Returns a Gaussian kernel using the specified cutoff standard deviation.
        #
        # PyTorch requires the kernel to be of a particular shape in order to apply
        # it to an image. Specifically, the kernel needs to be of shape (c, 1, k, k)
        # where c is the # channels in the image.
        #
        # Start by getting a 2D Gaussian kernel using your implementation from earlier,
        # which will be of shape (k, k). Then, let's say you have an RGB image, you
        # will need to turn this into a Tensor of shape (3, 1, k, k) by stacking the
        # Gaussian kernel 3 times.
        #
        # Args:
        #     cutoff_standarddeviation: int specifying the cutoff standard deviation
        # Returns:
        #     kernel: Tensor of shape (c, 1, k, k) where c is # channels
        #
        # HINTS:
        # - Since the # channels may differ across each image in the dataset, make
        #   sure you don't hardcode the dimensions you reshape the kernel to. There
        #   is a variable defined in this class to give you channel information.
        # - You can use torch.reshape() to change the dimensions of the tensor.
        # - You can use torch's repeat() to repeat a tensor along specified axes.

        kernel = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        #if type(cutoff_standarddeviation) is not int:
           # cutoff_standarddeviation = int(cutoff_standarddeviation[0])
        kernel = create_2d_gaussian_kernel(cutoff_standarddeviation)
        kernel = kernel.unsqueeze(0)

        kernel = kernel.unsqueeze(0)
        kernel = torch.cat((kernel, kernel, kernel), dim=0)
        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return kernel

    def low_pass(self, x, kernel):
        # Apply low pass filter to the input image.
        #
        # Args:
        #     x: Tensor of shape (b, c, m, n) where b is batch size
        #     kernel: low pass filter to be applied to the image
        # Returns:
        #     filtered_image: Tensor of shape (b, c, m, n)
        #
        # HINT:
        # - You should use the 2d convolution operator from torch.nn.functional.
        # - Make sure to pad the image appropriately (it's a parameter to the
        #   convolution function you should use here!).
        # - Pass self.n_channels as the value to the "groups" parameter of the
        #   convolution function. This represents the # of channels that the filter
        #   will be applied to.

        filtered_image = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

       # b, c, m, n = x.size()
        #k = kernel.size()[2]
        #pad = int(np.floor(k/2))

        
        #filtered_image = torch.nn.functional.conv2d(x, kernel, padding = pad, groups = self.n_channels)
        b, c, m, n = kernel.shape

        filtered_image = torch.nn.functional.conv2d(x, kernel, padding = (m // 2, n // 2), groups = self.n_channels)

        ############################    HINTS:#################################################
        #                             END OF YOUR CODE
        #############################################################################

        return filtered_image

    def forward(self, image1, image2, cutoff_standarddeviation):
        # Take two images and creates a hybrid image. Returns the low frequency
        # content of image1, the high frequency content of image 2, and the hybrid
        # image.
        #
        # Args:
        #     image1: Tensor of shape (b, m, n, c)
        #     image2: Tensor of shape (b, m, n, c)
        #     cutoff_standarddeviation: Tensor of shape (b)
        # Returns:
        #     low_frequencies: Tensor of shape (b, m, n, c)
        #     high_frequencies: Tensor of shape (b, m, n, c)
        #     hybrid_image: Tensor of shape (b, m, n, c)
        #
        # HINTS:
        # - You will use the get_kernel() function and your low_pass() function in
        #   this function.
        # - Don't forget to make sure to clip the pixel values >=0 and <=1. You can
        #   use torch.clamp().
        # - If you want to use images with different dimensions, you should resize
        #   them in the HybridImageDataset class using torchvision.transforms.

        self.n_channels = image1.shape[1]

        low_frequencies = torch.Tensor()
        high_frequencies = torch.Tensor()
        hybrid_image = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################

        kernel = self.get_kernel(cutoff_standarddeviation)
        low_frequencies = self.low_pass(image1, kernel)
        high_frequencies = image2 - self.low_pass(image2, kernel)
        hybrid_frequencies = low_frequencies + high_frequencies
        hybrid_image = torch.clamp(hybrid_frequencies, 0, 1)

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return low_frequencies, high_frequencies, hybrid_image


# TODO - 8
def my_median_filter(image: torch.FloatTensor, filter_size: Union[tuple, int]) -> torch.FloatTensor:
    """
    Apply a median filter to an image. Return the filtered image.
    Args
    - image: Torch tensor of shape (m, n, 1) or Torch tensor of shape (m, n).
    - filter: Torch tensor of shape (k, j). If an integer is passed then all dimensions
              are considered of the same size. Input will always have odd size.
    Returns
    - filtered_image: Torch tensor of shape (m, n, 1)

    - You may not use any libraries that do the work for you. Using torch to work
     with matrices is fine and encouraged. Using OpenCV/scipy or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take a long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Useful functions: torch.median and torch.nn.functional.pad
    """
    if len(image.size()) == 3:
        assert image.size()[2] == 1

    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert filter_size[0] % 2 == 1
    assert filter_size[1] % 2 == 1

    filtered_image = torch.Tensor()

    ############################################################################
    #                           TODO: YOUR CODE HERE
    ############################################################################

    row, col = image.size()
    krow, kcol = filter_size
    padrow = int(krow // 2)
    padcol = int(kcol // 2)
    new_img = torch.zeros(row + padrow * 2, col + padcol * 2)
    for i in range(padrow, row + padrow):
        for j in range(padcol, col + padcol):
            new_img[i][j] = image[i - padrow][j - padcol]

    filtered_image = torch.zeros(row, col)
    for i in range(row):
        for j in range(col):
            l = []
            for k in range(krow):
                line = new_img[i + k, j:j + kcol].tolist()
                l = l + line
                l.sort()
                median = l[len(l) // 2]
            filtered_image[i][j] = median
    filtered_image = filtered_image[:, :, None]
    
    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return filtered_image


#############################################################################
# Extra credit opportunity (for UNDERGRAD) below
#
# Note: This part is REQUIRED for GRAD students
#############################################################################

# Matrix multiplication helper
def complex_multiply_real(m1, m2):

    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2
    imag2 = torch.zeros(real2.shape)
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


def complex_multiply_complex(m1, m2):

    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2[:, :, 0]
    imag2 = m2[:, :, 1]
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


def dft_matrix(N):

    U = torch.Tensor()

    torch.pi = torch.acos(torch.zeros(1)).item() * \
        2  # which is 3.1415927410125732

    U = torch.zeros(N, N, 2)
    for i in range(N):
        for j in range(N):
            U[i, j, 0] = (1/N) * torch.cos(torch.tensor(2 * torch.pi * i * j) / N)
            U[i, j, 1] = (-1/N) * torch.sin(torch.tensor(2 * torch.pi * i * j) / N)

    return U


def my_dft(img):

    dft = torch.Tensor()

    assert img.shape[0] == img.shape[1], "Input image should be a square matrix"

    N = img.size()[0]
    U = dft_matrix(N)

    real = complex_multiply_real(U, img)
    dft = complex_multiply_complex(real, U)

    return dft


# TODO - 11
def dft_filter(img):
    # Take a square image as input, performs a low-pass filter and return the filtered image
    #
    # Args
    # - img: a 2D grayscale image whose width equals height, size: (N,N)
    # Returns
    # - img_back: the filtered image whose size is also (N,N)
    #
    # HINTS:
    # - You will need your implemented DFT filter for this function
    # - We don't care how much frequency you want to retain, if only it returns reasonable results
    # - Since you already implemented DFT part, you're allowed to use the torch.ifft in this part for convenience, though not necessary

    img_back = torch.Tensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################


    fre = my_dft(img)
    mask = torch.zeros(fre.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i + j < 2.5 * img.shape[0]:
                mask[i, j, :] = 1

    fre_shift = torch.ifft(fre * mask, 2, True)

    img_back = torch.sqrt(fre_shift[:, :, 0] ** 2 + fre_shift[:, :, 1] ** 2)

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return img_back
