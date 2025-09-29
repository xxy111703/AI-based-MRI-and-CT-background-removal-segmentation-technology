# AI-based-MRI-and-CT-background-removal-segmentation-technology
Two machine learning methods( GAC and the spectral clustering) in medical image segmentation
Medical Image Segmentation using Spectral Clustering and GAC


This repository contains the implementation and experimental results for the undergraduate thesis on medical image segmentation. The project explores, implements, and optimizes two advanced methods for foreground-background separation in medical images: Spectral Clustering and Morphological Geodesic Active Contours (GAC).

📖 Project Overview
Medical image segmentation is critically important for disease diagnosis and medical research. This project first reviews traditional segmentation techniques and then focuses on two powerful methods: Spectral Clustering and Morphological Geodesic Active Contours (GAC). The core of this work involves:


Implementing these two algorithms and demonstrating their feasibility on real-world medical images (brain CT/MRI scans).




Optimizing the key parameters for both methods to achieve the best possible segmentation results.




Analyzing and Comparing the results to showcase the impact of parameter tuning and the superiority of the optimized models.



✨ Key Features
This project provides an implementation of two distinct segmentation approaches:

1. Spectral Clustering for Segmentation

Methodology: This approach treats image segmentation as a graph partitioning problem. Each pixel is a node in a graph, and the edge weights are determined by the similarity between pixels (using a Gaussian kernel). Segmentation is achieved by clustering the eigenvectors (specifically the Fiedler vector) of the graph's normalized Laplacian matrix using K-Means.




Application: This method is particularly effective for capturing the overall shape and external contours of a target, such as identifying a tumor in a CT or MRI scan.

2. Morphological Geodesic Active Contours (GAC)

Methodology: GAC is a model-based technique that evolves an initial contour towards the boundaries of the target object by minimizing an energy function. It combines the flexibility of level-set methods with the robustness of morphological operations to handle complex shapes and topological changes.




Application: GAC is well-suited for scenarios requiring precise delineation of internal structures, such as distinguishing different stages of lesion tissue.

🚀 Parameter Optimization
A major contribution of this work is the systematic optimization of model parameters. Instead of manual tuning, we employed Bayesian optimization techniques to efficiently search the parameter space.

GAC Parameter Optimization

Objective Function: Maximizing the Dice Coefficient, a standard metric for comparing the similarity of two samples, to measure the overlap between the segmentation result and the ground truth mask.


Algorithm: Gaussian Process Optimization (GPO) was used to find the optimal parameters over 20 iterations.


Optimal Parameters Found:
smoothing: 14 
balloon: -1.00 
threshold: 0.60 


Best Performance: Achieved a maximum Dice Coefficient of 0.3687.

Spectral Clustering Parameter Optimization

Objective Function: Maximizing the Silhouette Coefficient, an unsupervised metric used to evaluate the quality of clustering by measuring the cohesion and separation of clusters.



Algorithm: A Bayesian optimization approach was used over 20 iterations to find the best parameter combination.

Optimal Parameters Found:
radius: 41 
sigma: 0.0973 

📊 Results
The experiments successfully demonstrate the feasibility of both methods. The parameter optimization process significantly improved segmentation accuracy compared to initial or arbitrary parameter sets.


Optimal GAC Segmentation Result
The result with optimized parameters shows a clear and smooth contour that accurately fits the target boundaries, avoiding over- or under-segmentation seen with other parameters.

(Image placeholder for Figure 4.2)

Optimal Spectral Clustering Result
The optimized parameters for spectral clustering led to a much cleaner separation of foreground and background compared to non-optimized attempts.

(Image placeholder for Figure 4.7)

🛠️ Installation & Usage
Prerequisites
Python 3.x

NumPy

scikit-image

scikit-learn

scikit-optimize

Matplotlib

Installation
Bash

git clone https://github.com/your-username/medical-image-segmentation.git
cd medical-image-segmentation
pip install -r requirements.txt
Running the Code
To run a segmentation experiment:

Bash

# Run GAC segmentation with default parameters
python run_segmentation.py --method GAC --image_path /path/to/your/image.png

# Run Spectral Clustering with specific parameters
python run_segmentation.py --method spectral --image_path /path/to/your/image.png --radius 41 --sigma 0.0973
To run the parameter optimization:

Bash

python optimize_parameters.py --method GAC
📁 Repository Structure
.
├── data/                  # Sample medical images
├── results/               # Folder to save segmentation outputs
├── src/
│   ├── gac.py             # Implementation of GAC
│   ├── spectral.py        # Implementation of Spectral Clustering
│   └── utils.py           # Utility functions for image processing
├── run_segmentation.py    # Main script to run experiments
├── optimize_parameters.py # Script for parameter optimization
├── requirements.txt       # Python dependencies
└── README.md
limitations and Future Work
Limitations
The experiments and optimizations were conducted on a limited number of brain CT images (two), which may not be sufficient for general conclusions.

The parameter search space was constrained due to computational time limits.

Future Work

Integration with Deep Learning: Combine these methods with deep learning to extract more representative features for clustering or to guide the GAC evolution.


Adaptive Parameter Tuning: Develop mechanisms for automatically adjusting parameters based on the characteristics of different input images.

📜 Citation
This work is based on the undergraduate thesis from Sun Yat-sen University. If you use this code or these findings in your research, please consider citing the original paper.
