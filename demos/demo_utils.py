"""
    Utils for demoing HAH networks in notebooks
"""
import numpy as np

def get_random_data(num_samples, output_dim, latent_dim=None):
    """Make a random high-dimensional dataset with low-dimensional
    latent structure
    
    Args:
        num_samples: the number of samples to return
        output_dim: the dimensionality of each sample
        latent_dim: the latent dimension of the data.
            set to None to do no projection to lower dim
        
    Returns:
        list of length num_samples the length output_dim numpy arrays
    """

    if latent_dim is not None and latent_dim > output_dim:
        raise ValueError("Latent dim must be <= output_dim")

    data_set = np.random.randn(num_samples, output_dim)

    if latent_dim:
        # Use top SVD to reconstruct dataset with only top
        # 'latent_dim' singular vectors
        data_set -= np.mean(data_set, axis=0)
        U, S, V = np.linalg.svd(data_set)
        data_set = (U[:, :latent_dim] * S[:latent_dim]).dot(V[:latent_dim, :])

    data_set -= np.mean(data_set, axis=0)
    return list(data_set)
