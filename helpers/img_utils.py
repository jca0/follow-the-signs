import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def img_to_occgrid(img_path, resolution=1):
    img = Image.open(img_path).convert("L")
    img_arr = np.array(img)

    occupancy = np.full(img_arr.shape, -1)
    # Treat black pixels as occupied (1) and white pixels as open (0)
    occupancy[img_arr < 128] = 1   # black (occupied)
    occupancy[img_arr >= 128] = 0  # white (open)

    occupied_coords = np.argwhere(occupancy == 1)
    y_min, x_min = occupied_coords.min(axis=0)
    y_max, x_max = occupied_coords.max(axis=0) + 1

    cropped_occupancy = occupancy[y_min:y_max, x_min:x_max]

    if resolution > 1:
        h, w = cropped_occupancy.shape
        h_res = h // resolution
        w_res = w // resolution
        downsampled_occupancy = np.full((h_res, w_res), -1)

        for i in range(h_res):
            for j in range(w_res):
                block = cropped_occupancy[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution]
                if np.any(block == 1):
                    downsampled_occupancy[i, j] = 1
                else:
                    downsampled_occupancy[i, j] = 0

    else:
        downsampled_occupancy = cropped_occupancy

    return downsampled_occupancy