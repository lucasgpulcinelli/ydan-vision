# YDAn-vision (Youtube Data Analysis, image processing focus)
YDAn-vision (pronounced like "you done") is a data engineering project about gathering youtube video and channel data recursively and then processing it to obtain meaningful information from thumbnails using unsupervised machine learning.

The project uses a classical (concatenated HOG and color histograms) and a vision transformer feature extractor (DINO), HDBSCAN or K-Means as a clustering method, PCA as an optional dimensionality reduction technique, and T-SNE as a data visualization method.

Because of the image volume, we needed to use a parallel processing library to scale feature extraction, clustering and visualization (we could have sampled the data, but wanted to learn about unsupervised learning at scale anyways), so we used the dask ecosystem and cuda-enabled libraries to do so.

## Made Fully By
- Lucas Eduardo Gulka Pulcinelli,
- Matheus Pereira Dias,
- Mateus Santos Messias.
