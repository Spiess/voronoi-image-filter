import random

import numpy as np


def poisson_disk(r, width, height, k=30):
    """
    Poisson-disc sampling according to https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf and
    similar to the JavaScript implementation from https://www.jasondavies.com/poisson-disc/.

    :param r: points are generated in the radius interval [r, 2r] from their neighbors (see paper)
    :param width: width of the domain (e.g. image)
    :param height: height of the domain (e.g. image)
    :param k: number of attempts to generate valid neighbors for each point (see paper)
    :return: numpy array of Poisson-disc sampled points
    """
    # Step 0: Initialization
    cell_size = r / np.sqrt(2)

    points = []
    grid = np.ones((np.ceil(height / cell_size).astype(np.int), np.ceil(width / cell_size).astype(np.int)), dtype=np.int) * -1
    active = []

    # Normalizing constant for annulus sampling
    r_sqr = r ** 2
    a = 3 * r_sqr

    def emit_sample(point):
        points.append(point)
        index = len(points) - 1
        int_point = (point // cell_size).astype(np.int)
        assert(grid[int_point[0], int_point[1]] == -1)
        grid[int_point[0], int_point[1]] = index
        active.append(index)

    def generate_around(point):
        theta = random.random() * 2 * np.pi
        radius = np.sqrt(random.random() * a + r_sqr)

        return point + np.array([radius * np.sin(theta), radius * np.cos(theta)])

    def check_extents(point):
        return 0 < point[0] < height and 0 < point[1] < width

    def check_neighborhood(point):
        iy, ix = (point // cell_size).astype(np.int)
        y0 = max(0, iy - 1)
        x0 = max(0, ix - 1)
        y1 = min(iy + 2, grid.shape[0])
        x1 = min(ix + 2, grid.shape[1])

        for ny in range(y0, y1):
            for nx in range(x0, x1):
                grid_index = grid[ny, nx]
                if grid_index == -1:
                    continue
                grid_point = points[grid_index]
                if np.square(grid_point - point).sum() < r_sqr:
                    return False

        return True

    def check_valid(point):
        return check_extents(point) and check_neighborhood(point)

    # Step 1: Initial sample
    emit_sample(np.random.random_sample((2,)) * np.array([height, width]))

    # Step 2: Sampling
    while len(active) > 0:
        i = active[random.randint(0, len(active) - 1)]

        candidate_found = False
        for j in range(k):
            new_point = generate_around(points[i])
            if check_valid(new_point):
                candidate_found = True
                emit_sample(new_point)
                break

        if not candidate_found:
            active.remove(i)

    return np.array(points)
