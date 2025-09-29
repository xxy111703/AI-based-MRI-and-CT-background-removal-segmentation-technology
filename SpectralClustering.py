import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform
from skimage.util import img_as_float
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage.filters import gaussian  # For denoising
from skimage.morphology import closing, square  # For postprocessing
from skimage.io import imread
from skimage.color import rgb2gray



def build_affinity_matrix(image, radius=3, sigma=0.1):
    """
    Build a sparse affinity matrix for a 2D image.
    Only pixels within a disk of given radius (around each pixel) are connected.

    Parameters:
        image  : 2D NumPy array of shape (H, W)
        radius : int, the neighborhood radius (in pixels)
        sigma  : float, parameter for the Gaussian kernel based on intensity differences

    Returns:
        A      : sparse CSR matrix of shape (H*W, H*W)
    """
    H, W = image.shape
    n = H * W
    rows = []
    cols = []
    vals = []

    # Precompute neighbor offsets within a disk of the specified radius
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue  # skip the center pixel
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dy, dx))

    # Loop over every pixel and connect it with its local neighbors.
    for i in range(H):
        for j in range(W):
            idx = i * W + j  # linear index for pixel (i, j)
            I_p = image[i, j]
            for (dy, dx) in offsets:
                ni = i + dy
                nj = j + dx
                # Check for valid neighbor coordinates.
                if 0 <= ni < H and 0 <= nj < W:
                    nidx = ni * W + nj
                    I_q = image[ni, nj]
                    # Compute similarity using a Gaussian kernel based on intensity difference.
                    diff = I_p - I_q
                    affinity = np.exp(- (diff * diff) / (2 * sigma * sigma))
                    rows.append(idx)
                    cols.append(nidx)
                    vals.append(affinity)

    # Build a sparse matrix in COO format and then make it symmetric.
    A = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    A = (A + A.T) / 2.0  # ensure symmetry
    return A.tocsr()


def spectral_segmentation(image, radius=3, sigma=0.1):
    """
    Segment a 2D image into foreground and background using spectral clustering.

    The steps are:
      1. Build a locally sparse affinity matrix.
      2. Construct the symmetric normalized Laplacian.
      3. Compute the eigenvectors (using eigsh) and extract the Fiedler vector.
      4. Cluster the pixels into two groups with k-means.
      5. Decide which cluster is foreground based on average intensity.

    Parameters:
        image  : 2D NumPy array (256x256) with intensity values in [0,1].
        radius : Neighborhood radius for constructing the affinity matrix.
        sigma  : Gaussian kernel width for intensity differences.

    Returns:
        segmentation : 2D binary mask (uint8) of the segmentation.
    """
    H, W = image.shape
    n = H * W

    # 1. Build the sparse affinity matrix
    A = build_affinity_matrix(image, radius=radius, sigma=sigma)

    # 2. Compute the degree and the normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(d)
    np.divide(1, np.sqrt(d), out=d_inv_sqrt, where=(d > 0))

    D_inv_sqrt = diags(d_inv_sqrt)
    I = identity(n, format='csr', dtype=np.float32)
    L_sym = I - D_inv_sqrt @ A @ D_inv_sqrt

    # 3. Compute the eigen-decomposition (we need the second smallest eigenvector, the Fiedler vector)
    #    For two clusters, we request k=2 eigenpairs. The first eigenvector is constant.
    eigenvalues, eigenvectors = eigsh(L_sym, k=2, which='SM')
    # Extract the Fiedler vector (second eigenvector)
    fiedler_vector = eigenvectors[:, 1].reshape(-1, 1)

    # 4. Cluster pixels using k-means (clustering in 1D using the Fiedler vector)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(fiedler_vector)
    labels_image = labels.reshape(H, W)

    # 5. Decide which cluster corresponds to the foreground.
    #     (Assuming that the foreground is the brighter region.)
    mean0 = np.mean(image[labels_image == 0])
    mean1 = np.mean(image[labels_image == 1])
    foreground_label = 0 if mean0 > mean1 else 1
    segmentation = (labels_image == foreground_label).astype(np.uint8)

    return segmentation



def main():
    image = imread(r"C:\Users\HUAWEI\Desktop\毕业设计\MIT\Brain_Tumor_Detection\no\No14.jpg")
    image = rgb2gray(image)  # 转为灰度图，防止 shape 报错
    image = transform.resize(image, (128, 128), anti_aliasing=False)
    image = img_as_float(image)

    # 1. Denoising (Gaussian blur)
    image_denoised = gaussian(image, sigma=2.5)  # Adjust sigma as needed

    # 2. Spectral Segmentation (with potentially tuned parameters)
    segmentation = spectral_segmentation(image, radius=50, sigma=0.15)

    # 3. Postprocessing (Morphological closing)
    segmentation_closed = closing(segmentation, square(2))  # adjust square size as needed

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image_denoised, cmap=plt.cm.gray)
    axes[1].set_title("Denoised Image")
    axes[1].axis("off")

    axes[2].imshow(segmentation_closed, cmap=plt.cm.gray)
    axes[2].set_title("Segmented (and Postprocessed)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()