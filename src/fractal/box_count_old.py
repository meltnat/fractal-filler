"""This file is created to test the BoxCountOld class."""

import numpy as np


class BoxCountOld:
    """This class is created to test the BoxCountOld class."""
    def count(self, image: np.ndarray) -> float:
        """This function is created to test the BoxCountOld class."""
        x, y = image.shape
        boxsize = 1
        boxsizes = []
        counts = []
        while boxsize < x // 10:
            boxsize = boxsize * 2
            boxsizes.append(boxsize)
            count = 0
            for i in range(x // boxsize):
                for j in range(y // boxsize):
                    if np.sum(image[i * boxsize : (i + 1) * boxsize, j * boxsize : (j + 1) * boxsize]) > 0:
                        count += 1
            counts.append(count)

        return np.polyfit(np.log(boxsizes), np.log(counts), 1)[0]
