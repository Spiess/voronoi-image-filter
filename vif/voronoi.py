import numpy as np


def voronoi(points, image):
    """
    Generates Voronoi filtered image from the given Voronoi centers and image.

    :param points: numpy array of 2D points.
    :param image: numpy source image
    :return: Voronoi filtered image
    """
    group_map = np.ones((image.shape[0], image.shape[1]), dtype=np.int32) * -1

    for y in range(group_map.shape[0]):
        for x in range(group_map.shape[1]):
            distances = np.square(points - np.array([y, x])).sum(axis=1)
            group_map[y, x] = np.argmin(distances)

    voronoi_image = np.zeros_like(image)

    for i in range(len(points)):
        mask = group_map == i
        mean_color = image[mask].sum(axis=0) / mask.sum()
        voronoi_image[mask] = mean_color

    return voronoi_image
