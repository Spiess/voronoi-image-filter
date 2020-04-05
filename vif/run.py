import argparse

import numpy as np
from PIL import Image

from vif.sampling import poisson_disk
from vif.voronoi import voronoi


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('image', help='Path to image file.', type=str)
    parser.add_argument('--point-radius', help='Point radius for Poisson-disc sampling.', type=float)

    args = parser.parse_args()

    img = Image.open(args.image)
    img.load()
    data = np.asarray(img, dtype=np.uint8)

    width = data.shape[1]
    height = data.shape[0]

    r = args.point_radius if args.point_radius else width / 10

    # Extremely basic point generation method
    #
    # width_points = 10
    # height_points = 10
    #
    # x_step = width / width_points
    # x_start = x_step / 2
    #
    # y_step = height / height_points
    # y_start = y_step / 2
    #
    # points = np.mgrid[y_start:height:y_step, x_start:width:x_step]
    # points = np.transpose(points, (1, 2, 0)).reshape((-1, 2))
    #
    # points += np.random.standard_normal(points.shape) * np.array([y_start, x_start])

    points = poisson_disk(r, width, height)

    result = voronoi(points, data)

    nimg = Image.fromarray(result)
    nimg.show()


if __name__ == '__main__':
    main()
