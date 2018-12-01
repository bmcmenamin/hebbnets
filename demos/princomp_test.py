"""
    Test princomp extraction from CLI
"""

import argparse
import os
import sys

repo_root_path = os.path.abspath(os.path.join(os.path.pardir, 'hebbnets'))
if repo_root_path not in sys.path:
    sys.path.append(repo_root_path)

import numpy as np

from demo_utils import get_random_data
from hebbnets.hebbnets.hah_model import MultilayerHahNetwork


np.set_printoptions(suppress=True)


def _argparse():

    parser = argparse.ArgumentParser(
        prog="Testing HebbNet principal components",
        description="Testing HebbNet principal components by decomposing random data"
    )

    parser.add_argument(
        "--num_samples",
        help="Number of samples for synthetic data",
        default=25,
        type=int,
        required=False
    )

    parser.add_argument(
        "--data_dimension",
        help="Dimension of synthetic data",
        default=100,
        type=int,
        required=False
    )

    parser.add_argument(
        "--data_latent_dimension",
        help="Latent dimension of synthetic data",
        default=3,
        type=int,
        required=False
    )

    parser.add_argument(
        "--num_pc",
        help="Number of principle components to extract",
        default=2,
        type=int,
        required=False
    )

    return parser.parse_args()


def get_top_princomps(data_array, num_pcs):
    U, S, V = np.linalg.svd(np.array(data_array))
    _idx = np.argsort(S)[-num_pcs:]
    return V[_idx, :].T


def main(args):

    # Make data
    demo_data = get_random_data(
        args.num_samples,
        args.data_dimension,
        latent_dim=args.data_latent_dimension
    )

    # Build/train network
    hah_network = MultilayerHahNetwork(
        args.data_dimension,
        [args.num_pc],
        has_bias=False,
        act_type='linear',
    )

    hah_network.train(demo_data, num_epochs=1000)

    # Build/train network
    real_princomps = get_top_princomps(demo_data, args.num_pc)
    hebb_princomps = np.squeeze(hah_network.layers[0].input_weights)
    hebb_princomps /= np.linalg.norm(hebb_princomps, axis=0, keepdims=True)

    # Show the inner product of top two PCs with learned input weights
    inner_prod_mat = real_princomps.T.dot(hebb_princomps)

    prod_as_string = np.array_str(
        inner_prod_mat,
        suppress_small=True,
        precision=4
    )

    print(np.array_str(inner_prod_mat, precision=4))


if __name__ == "__main__":
    args = _argparse()
    main(args)
