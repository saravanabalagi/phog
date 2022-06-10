import cv2
import math
import numpy as np
import argparse
from numba import jit


def printMatDetails(m, desc="matrix"):
    print(f'{desc}: {m.dtype} {m.shape} [{m.min()}, {m.max()}], u: {m.mean()}, o: {m.std()}')


@jit
def getHistogram(edges, ors, mag, startX, startY, width, height, nbins):
    hist = np.zeros(nbins)
    for x in range(startX, startX + height):
        for y in range(startY, startY + width):
            if edges[x, y] > 0:
                bin = math.floor(ors[x, y])
                hist[bin] = hist[bin] + mag[x, y]
    return hist


def compute_phog(img, nbins):
    # Determine desc size
    desc_size = nbins + 4 * nbins + 16 * nbins

    # Convert the image to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # printMatDetails(img, 'Image BW')

    # Reduce noise, Apply Canny Edge Detector
    mean = np.mean(img)
    edges = cv2.blur(img, (3,3))
    # printMatDetails(edges, 'Image Blur')
    edges = cv2.Canny(edges, 0.66 * mean, 1.33 * mean)

    # printMatDetails(edges, 'edges')

    # Computing the gradients.
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Total Gradient (approximate)
    grad_m = np.abs(grad_x) + np.abs(grad_y)

    # printMatDetails(grad_x, 'grad_x')
    # printMatDetails(grad_y, 'grad_y')
    # printMatDetails(grad_m, 'grad_m')

    # Computing orientations
    grad_o = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # printMatDetails(grad_o, 'grad_o')

    # Quantizing orientations into bins.
    w = 360.0 / nbins
    grad_o = grad_o / w

    # printMatDetails(grad_o, 'grad_o')

    # Creating the descriptor.
    desc = np.zeros(desc_size, dtype=np.float32)
    width = img.shape[1]
    height = img.shape[0]

    # Level 0
    desc[0:nbins] = getHistogram(edges, grad_o, grad_m, 0, 0, width, height, nbins)
    # cv2.imshow('image', img)

    # Level 1
    desc[nbins:nbins*2] = getHistogram(edges, grad_o, grad_m, 0, 0, width // 2, height // 2, nbins)
    desc[nbins*2:nbins*3] = getHistogram(edges, grad_o, grad_m, 0, width // 2, width // 2, height // 2, nbins)
    desc[nbins*3:nbins*4] = getHistogram(edges, grad_o, grad_m, height // 2, 0, width // 2, height // 2, nbins)
    desc[nbins*4:nbins*5] = getHistogram(edges, grad_o, grad_m, height // 2, width // 2, width // 2, height // 2, nbins)

    # Level 2
    wstep = width // 4
    hstep = height // 4
    binPos = 5 # Next free section in the histogram
    for i in range(4):
        for j in range(4):
            desc[nbins*binPos:nbins*(binPos+1)] = getHistogram(edges, grad_o, grad_m, i * hstep, j * wstep, wstep, hstep, nbins)
            binPos += 1

    # Normalizing the histogram
    desc = desc / desc.sum()

    return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help='Path to img')
    parser.add_argument('--nbins', type=int, default=60, help='Number of orientation bins')
    parser.add_argument('--pyramid_height', type=int, default=2, help='Number of levels in the pyramid')
    args = parser.parse_args()

    # Read image
    img = cv2.imread(args.img_path)
    printMatDetails(img, 'Image')

    desc = compute_phog(img, nbins=args.nbins)
    printMatDetails(desc, 'Gdsc')

    filename = 'data/desc_py.txt'
    np.savetxt(filename, desc)
    print(f'Successfully saved {filename}')


if __name__ == '__main__':
    main()
