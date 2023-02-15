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


def create_2d_gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:

    kernel_2d = torch.Tensor()

    standard_deviation = int(standard_deviation)
    k = 4 * standard_deviation + 1
    mean = k // 2
    kernel_2d = torch.arange(-mean, mean + 1).reshape((-1, 1)) ** 2
    kernel_2d = kernel_2d + kernel_2d.transpose(1, 0)
    kernel_2d = torch.exp(-0.5 * (kernel_2d - mean) / standard_deviation ** 2)
    sum = torch.sum(kernel_2d)
    kernel_2d /= sum

    return kernel_2d


def my_imfilter(image, image_filter, image_name="Image"):

    filtered_image = torch.Tensor()

    assert image_filter.shape[0] % 2 == 1
    assert image_filter.shape[1] % 2 == 1

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

    return filtered_image

def create_hybrid_image(image1, image2, filter):

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

    low_freq_image = my_imfilter(image1, filter)
    low_freq_image2 = my_imfilter(image2, filter)

    high_freq_image = image2 - low_freq_image2

    hybrid_image = torch.clamp( (low_freq_image + high_freq_image), 0, 1)


    return low_freq_image, high_freq_image, hybrid_image


def make_dataset(path: str) -> Tuple[List[str], List[str]]:

    images_a = []
    images_b = []

    file_list = os.listdir(path)
    for fname in file_list:
        if 'a_' in fname:
            images_a.append(path + fname)
        elif 'b_' in fname:
            images_b.append(path + fname)

    images_a.sort()
    images_b.sort()

    return images_a, images_b


def get_cutoff_standardddeviations(path: str) -> List[int]:
    cutoffs = []

    with open(path, "r") as f:
      cutoffs = f.read().split('\n')
    for i in cutoffs[::-1]:
      if i == '':
        cutoffs.pop(-1)
      else:
        break
    cutoffs = list(map(int, cutoffs))

    return cutoffs


class HybridImageDataset(data.Dataset):
    def __init__(self, image_dir: str, cf_file: str) -> None:
        images_a, images_b = make_dataset(image_dir)

        self.cutoffs = get_cutoff_standardddeviations(cf_file)

        self.transform = torchvision.transforms.ToTensor()


        self.images_a = images_a
        self.images_b = images_b

    def __len__(self) -> int:

        return len(self.images_a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image_a = torch.Tensor()
        image_b = torch.Tensor()
        cutoff = 0
        image_a = PIL.Image.open(self.images_a[idx] )
        image_a = self.transform(image_a)
        image_b = PIL.Image.open(self.images_b[idx] )
        image_b = self.transform(image_b)
        cutoff = self.cutoffs[idx]

        return image_a, image_b, cutoff


class HybridImageModel(nn.Module):
    def __init__(self):
        # Initializes an instance of the HybridImageModel class.
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_standarddeviation: int) -> torch.Tensor:

        kernel = torch.Tensor()
        kernel = create_2d_gaussian_kernel(cutoff_standarddeviation)
        kernel = kernel.unsqueeze(0)

        kernel = kernel.unsqueeze(0)
        kernel = torch.cat((kernel, kernel, kernel), dim=0)


        return kernel

    def low_pass(self, x, kernel):

        filtered_image = torch.Tensor()

        b, c, m, n = kernel.shape

        filtered_image = torch.nn.functional.conv2d(x, kernel, padding = (m // 2, n // 2), groups = self.n_channels)

        return filtered_image

    def forward(self, image1, image2, cutoff_standarddeviation):

        self.n_channels = image1.shape[1]

        low_frequencies = torch.Tensor()
        high_frequencies = torch.Tensor()
        hybrid_image = torch.Tensor()
        kernel = self.get_kernel(cutoff_standarddeviation)
        low_frequencies = self.low_pass(image1, kernel)
        high_frequencies = image2 - self.low_pass(image2, kernel)
        hybrid_frequencies = low_frequencies + high_frequencies
        hybrid_image = torch.clamp(hybrid_frequencies, 0, 1)


        return low_frequencies, high_frequencies, hybrid_image


def my_median_filter(image: torch.FloatTensor, filter_size: Union[tuple, int]) -> torch.FloatTensor:

    if len(image.size()) == 3:
        assert image.size()[2] == 1

    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert filter_size[0] % 2 == 1
    assert filter_size[1] % 2 == 1

    filtered_image = torch.Tensor()

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

    return filtered_image


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

def dft_filter(img):
    img_back = torch.Tensor()
    fre = my_dft(img)
    mask = torch.zeros(fre.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i + j < 2.5 * img.shape[0]:
                mask[i, j, :] = 1

    fre_shift = torch.ifft(fre * mask, 2, True)

    img_back = torch.sqrt(fre_shift[:, :, 0] ** 2 + fre_shift[:, :, 1] ** 2)

    return img_back
