import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.io import imread
from skimage import util, segmentation, color
from tqdm import tqdm


def store_evolution_in(lst):
    """Callback to store the evolution of the level set during segmentation."""

    def _store(x):
        lst.append(np.copy(x))

    return _store


def load_image(filepath):
    """Load an image and convert to grayscale if necessary."""
    image = util.img_as_float(imread(filepath))
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return image


def segment_image(image, num_iter=300, smoothing=10, balloon=-0.95, threshold=0.7):
    """Perform morphological geodesic active contour segmentation."""
    gimage = segmentation.inverse_gaussian_gradient(image)

    # Create an initial level set (simple rectangle)
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1

    # Store evolution for visualization
    evolution = []
    callback = store_evolution_in(evolution)

    # Define progress bar
    with tqdm(total=num_iter, desc="Segmentation Progress") as pbar:
        def progress_callback(x):
            pbar.update(1)
            callback(x)

        level_set = segmentation.morphological_geodesic_active_contour(
            gimage,
            num_iter=num_iter,
            init_level_set=init_ls,
            smoothing=smoothing,
            balloon=balloon,
            threshold=threshold,
            iter_callback=progress_callback
        )

    return level_set, evolution


def plot_results(image, level_set, smoothing, balloon, threshold):
    """Plot the original image, level set, and segmentation contour."""
    foreground_mask = level_set > 0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes.ravel()

    # 1. Original Image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # 2. Final Level Set
    ax[1].imshow(level_set, cmap='gray')
    ax[1].set_title(f"Level Set (smoothing={smoothing}, balloon={balloon}, threshold={threshold})")
    ax[1].axis('off')

    # 3. Foreground Mask Overlay
    ax[2].imshow(image, cmap='gray')
    ax[2].contour(foreground_mask, [0.5], colors='r', linewidths=2)
    ax[2].set_title("Foreground Mask Overlay")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    image_path = "/Users/xinry/Desktop/毕业设计/python-XIN RUNYAN/毕业设计脑ct图.jpg"
    image = load_image(image_path)

    # Iterate over the parameter ranges
    for smoothing in range(13, 17):  # smoothing from 13 to 16
        for balloon in np.arange(-1, -0.7, 0.1):  # balloon from -1 to -0.8
            threshold = 0.7  # threshold from 0.5 to 0.8
            print(f"Processing smoothing={smoothing}, balloon={balloon}, threshold={threshold}")

            # Perform segmentation with current parameters
            level_set, evolution = segment_image(image, num_iter=300, smoothing=smoothing, balloon=balloon,
                                                 threshold=threshold)

            # Plot results with current parameter values
            plot_results(image, level_set, smoothing, balloon, threshold)


if __name__ == "__main__":
    main()