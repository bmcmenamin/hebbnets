"""
    Utils for demoing HAH networks in notebooks
"""
import itertools
import random
import requests
from io import BytesIO

import numpy as np
from PIL import Image

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


def retry_get_request(url, max_retries):
    """Run a get request with multiple retries"""

    response = requests.Response()
    num_retries = 0
    while num_retries < max_retries:
        try:
            response = requests.get(url)
        except:
            pass

        if response.status_code == 200:
            return response

        print("Bad status code ({}). Retrying.".format(response.status_code))
        num_retries += 1
    
    return response

def yield_images_from_urls(url_list, max_retries=5):
    """Given a list of image URLs, yield each as a PIL image
    Args:
        url_list: list of URLs to pull
        max_retries: max number of retries to downlaod image
    Yields:
        PIL images
    """
    for url in url_list:
        response = retry_get_request(url, max_retries)

        if response.status_code != 200:
            print('No response for {}. Skipping.'.format(url))
        else:
            img_bytes = BytesIO(response.content)
            try:
                pil_img = Image.open(img_bytes)
                yield pil_img
            except OSError:
                print("Can't interpret {} as image. Skipping.".format(url))

def yield_image_patches(input_image, patch_size, patch_stride):
    """Yield image patches from input PIL image
    Args:
        input_image: PIL image
        patch_size: tuple of width/height for patches
        patch_string: tuple of stride size for width/height
    Yields
        numpy array of pixel intensities of size patch_size
    """
    
    grey_image = input_image.convert('L')

    patch_width, patch_height = patch_size
    stride_width, stride_height = patch_stride
    img_width, img_height = grey_image.size

    left_coord = range(0, img_width - patch_width, stride_width)
    top_coord =  range(0, img_height - patch_height, stride_height)

    bbox_list = [
        (left, top, left + patch_width, top + patch_height)
        for left, top in itertools.product(left_coord, top_coord)
    ]
    random.shuffle(bbox_list)

    for bbox in bbox_list:
        yield np.array(grey_image.crop(bbox))


def yeild_patch_batch(image_urls, patch_size=[21, 21], patch_stride=[15, 15], batch_size=1000):
    """Get batches of image patches from images loaded via URL
    Args:
        image_urls: list of urls of images
        patch_size: tuple of width/height for each patsh
        patch_stride: tuple of height/width for patch strides
        batch_size: max number of images in each batch
    """

    patch_batch = []
    for pil_image in yield_images_from_urls(image_urls):
        for patch in yield_image_patches(pil_image, patch_size, patch_stride):
            patch_batch.append(patch)
            if len(patch_batch) >= batch_size:
                yield patch_batch
                patch_batch = []    
    return patch_batch

def turn_input_weights_to_pilimg(hah_layer, patch_size):
    """Turn input weights into PIL images
    Args:
        hah_layer: hah layer object
    Returns:
        list of PIL images
    """
    pil_imgs = []
    
    weights = hah_layer.input_weights
    if hah_layer.params['bias']:
        weights = weights[:-1, :]

    for node_num in range(weights.shape[1]):
        np_img = weights[:, node_num].reshape(patch_size)
        pil_imgs.append(Image.fromarray(np_img, mode='L'))
    
    return pil_imgs

def place_pilimgs_in_grid(list_of_imgs, pixel_buffer=10):
    """Put list of PIL images into a single grid image
    
    Args:
        list_of_imgs: list of uniformly-sized PIL images
        pixel_buffer: number of pixels to put between images
    Return:
        PIL image of everythin
    """
    num_pics = len(list_of_imgs)
    grid_height = int(np.sqrt(num_pics))
    grid_width = int(np.ceil(num_pics / grid_height))

    size_set = set(p.size for p in list_of_imgs)
    if len(size_set) > 1:
        raise ValueError("Not all images are the same size")
    patch_size = size_set.pop()

    pix_width = grid_width * patch_size[0] + (grid_width - 1) * pixel_buffer
    pix_height = grid_height * patch_size[1] + (grid_height - 1) * pixel_buffer
    
    canvas = Image.new('RGB', (pix_width, pix_height), (255, 255, 255))
    for nw, nh in itertools.product(range(grid_width), range(grid_height)):
        if list_of_imgs:
            canvas.paste(
                list_of_imgs.pop(),
                (
                    nw * (patch_size[0] + pixel_buffer),
                    nh * (patch_size[1] + pixel_buffer)
                )
            )
    return canvas
