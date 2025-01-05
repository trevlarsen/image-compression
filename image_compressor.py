import numpy as np
from scipy.linalg import svd
from matplotlib import pyplot as plt
from imageio.v2 import imread

class ImageCompressor:
    """A class for compressing images using the Singular Value Decomposition (SVD).

    This class provides functionality to compress grayscale and color images by approximating
    them with a lower rank representation.

    Methods:
        compress_image: Compress an image to a specified rank and visualize the results.
    """

    def __init__(self, filename):
        """Initialize the ImageCompressor with the specified image.

        Args:
            filename (str): Path to the image file.
        """
        self.image = imread(filename) / 255.0  # Normalize image to range [0, 1].
        self.is_color = self.image.ndim == 3

        # Keep track of how many entries we save in the compressed version.
        self.entries_saved = 0

    def compress_image(self, rank):
        """Compress the image to the specified rank and display the result.

        Args:
            rank (int): Rank of the desired approximation.
        """
        # Optionally clamp the rank to the minimum dimension of the image
        if self.is_color:
            height, width, _ = self.image.shape
        else:
            height, width = self.image.shape
        rank = min(rank, min(height, width))

        if rank < 1:
            raise ValueError("Rank must be at least 1.")

        if self.is_color:
            # For color images, process each channel independently.
            compressed_image = self._compress_color_image(rank)
        else:
            # For grayscale images, process the single channel.
            compressed_image = self._compress_grayscale_image(rank)

        # Display the original and compressed images side by side.
        self._plot_comparison(compressed_image, rank)

    def _compress_grayscale_image(self, rank):
        """Compress a grayscale image to the specified rank.

        Args:
            rank (int): Rank of the desired approximation.

        Returns:
            ndarray: The compressed grayscale image.
        """
        approx, entries_used = self._svd_approximation(self.image, rank)
        # The total number of entries is image.size (height * width).
        self.entries_saved = self.image.size - entries_used
        return np.clip(approx, 0, 1)

    def _compress_color_image(self, rank):
        """Compress a color image to the specified rank.

        Args:
            rank (int): Rank of the desired approximation.

        Returns:
            ndarray: The compressed color image.
        """
        compressed_data = []
        for channel_idx in range(3):  # R, G, B
            approx, entries_used = self._svd_approximation(self.image[:, :, channel_idx], rank)
            compressed_data.append((np.clip(approx, 0, 1), entries_used))

        # Separate the approximations and entries counts
        compressed_channels = [data[0] for data in compressed_data]
        total_entries_used = sum(data[1] for data in compressed_data)

        # The total number of entries is image.size (height * width * 3 for color).
        self.entries_saved = self.image.size - total_entries_used

        return np.dstack(compressed_channels)

    def _svd_approximation(self, matrix, rank):
        """Perform a rank-k approximation of a matrix using SVD.

        Args:
            matrix (ndarray): The matrix to approximate.
            rank (int): Rank of the desired approximation.

        Returns:
            tuple:
                - ndarray: The rank-k approximation of the matrix.
                - int: Number of entries used in the truncated SVD representation.
        """
        U, S, VT = svd(matrix, full_matrices=False)
        U_k = U[:, :rank]
        S_k = np.diag(S[:rank])
        VT_k = VT[:rank, :]

        # The rank-k approximation
        approx = U_k @ S_k @ VT_k

        # The total entries used by the truncated SVD
        entries_used = U_k.size + S_k.size + VT_k.size

        return approx, entries_used

    def _plot_comparison(self, compressed_image, rank):
        """Plot the original image and the compressed image side by side.

        Args:
            compressed_image (ndarray): The compressed image to display.
            rank (int): The rank of the compressed image.
        """
        original_entries = self.image.size
        compressed_entries = original_entries - self.entries_saved
        percentage_reduction = (self.entries_saved / original_entries) * 100

        plt.figure(figsize=(10, 5))

        # Original image (with entry count)
        plt.subplot(1, 2, 1)
        plt.title(f"Original\n({original_entries} entries)")
        plt.imshow(self.image, cmap=None if self.is_color else 'gray')
        plt.axis('off')

        # Compressed image (with entry count)
        plt.subplot(1, 2, 2)
        plt.title(f"Compressed (Rank {rank})\n({compressed_entries} entries)")
        plt.imshow(compressed_image, cmap=None if self.is_color else 'gray')
        plt.axis('off')

        # Figure title: total entries saved and percentage reduction
        plt.suptitle(f"{self.entries_saved} entries saved "
                     f"({percentage_reduction:.2f}% reduction)")

        plt.tight_layout()
        # plt.savefig(f'demo_{rank}')
        plt.show()


if __name__ == "__main__":
    compressor = ImageCompressor("images/hubble.jpg")
    compressor.compress_image(20)
    compressor.compress_image(50)
    compressor.compress_image(100)

