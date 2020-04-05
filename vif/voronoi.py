import numpy as np


def print_progress(i, iterations, size=20):
    """
    Prints a progress bar.

    The progress bar has 'size' number of bars and shows the progress of 'i' relative to the total number of
    'iterations'.

    :param i: the current iteration
    :param iterations: the total number of iterations
    :param size: the number of bars to print when the progress bar is full
    """
    completion = (i + 1) / iterations
    fraction = int(completion * size)
    print(f'\r[{("=" * fraction)}{(" " * (size - fraction))}] {completion:.0%}', end='')


def voronoi(points, image, verbose=True, dtype=np.float32, memory_save_mode=False):
    """
    Generates Voronoi filtered image from the given Voronoi centers and image.

    :param points: numpy array of 2D points.
    :param image: numpy source image
    :param verbose: print progress
    :param dtype: data type to work with (adjust for potentially more accurate results)
    :param memory_save_mode: run in memory save mode increasing computation time, unless else memory is insufficient
    :return: Voronoi filtered image
    """
    points = points.astype(dtype)

    if memory_save_mode:
        print('Calculating groups...')
        group_map = np.ones((image.shape[0], image.shape[1]), dtype=np.int32) * -1

        pixels = image.shape[0] * image.shape[1]
        if verbose:
            print_progress(-1, pixels)
        for y in range(group_map.shape[0]):
            for x in range(group_map.shape[1]):
                distances = np.square(points - np.array([y, x])).sum(axis=1)
                group_map[y, x] = np.argmin(distances)
                if verbose:
                    print_progress(y * image.shape[1] + x, pixels)

        if verbose:
            print()
            print('Done!')
    else:
        if verbose:
            print('Generating grid...', end='')
        group_map = np.mgrid[:image.shape[0], :image.shape[1]].astype(dtype)
        if verbose:
            print(' done!')
            print('Transposing...', end='')
        group_map = group_map.transpose((1, 2, 0))
        if verbose:
            print(' done!')
            print('Expanding dimensions...', end='')
        group_map = np.expand_dims(group_map, axis=2)
        if verbose:
            print(' done!')
            print('Calculating differences...', end='')
        group_map = group_map - points
        if verbose:
            print(' done!')
            print('Calculating distances...', end='')
        group_map = np.square(group_map).sum(axis=-1)
        if verbose:
            print(' done!')
            print('Calculating groups...', end='')
        group_map = np.argmin(group_map, axis=-1)
        if verbose:
            print(' done!')

    if verbose:
        print('Calculating colors...', end='')
    voronoi_image = np.zeros_like(image)

    for i in range(len(points)):
        mask = group_map == i
        mean_color = image[mask].sum(axis=0) / mask.sum()
        voronoi_image[mask] = mean_color

    if verbose:
        print(' done!')

    return voronoi_image
