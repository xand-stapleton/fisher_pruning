''' 
Find the Fisher metric of the NN (KFAC). Diagonalise, and find reduced
network coordinates
'''

from collections import namedtuple
from nngeometry.metrics import FIM
from nngeometry.object.pspace import PMatBlockDiag, PMatDense, PMatKFAC, PMatDiag

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

def extract_blocks(A):
    '''
    Extract diagonal blocks from a symmetric matrix.

    Given a symmetric matrix `A`, this function extracts and returns a list of
    diagonal blocks. Each diagonal block is a square submatrix of `A` formed by
    non-zero elements in a contiguous diagonal pattern.

    Args:
        A (numpy.ndarray): A symmetric matrix as a 2D numpy array.

    Returns:
        list: A list containing diagonal blocks extracted from the matrix `A`.

    Raises:
        AssertionError: If the input matrix `A` is not square.

    Credits:
        Based on work of the GitHub user 'turingbirds'.

    Note:
        The matrix `A` is made symmetric by adding its transpose, allowing for
        extraction of diagonal blocks from the upper triangular part.
    '''

    assert A.shape[0] == A.shape[1], 'matrix A should be square'

    N = A.shape[0]

    # Make the matrix symmetric so we only have to check one triangle
    A_mirrored = A + A.T

    blocks = []
    start = 0
    blocksize = 0

    while start < N:
        blocksize += 1

        if np.all(A_mirrored[start:start + blocksize, start + blocksize:N] == 0):
            block = A[start:start + blocksize, start:start + blocksize]
            blocks.append(block)
            start += blocksize
            blocksize = 0

    return blocks

def find_fisher(net, loader, n_output, histogram=False,
                hist_file='fim_histo.pdf'):
    # Print the model (diagnostic)
    print(net)

    fisher_metric = FIM(net, loader,
                        representation=PMatDiag,
                        n_output=n_output,
                        variant='regression',
                        device='cpu')

    dense_metric = fisher_metric.get_diag()

    if histogram:
        plt.hist(dense_metric, bins=10, edgecolor='black')
        plt.xlabel('Fisher metric diagonal value')
        plt.ylabel('Frequency')
        plt.title('Fisher metric diagonal element frequencies')
        plt.savefig(hist_file)
        plt.show()

    return dense_metric

def diagonalise_fisher(fim, num_stiff=120):
    '''Calculate the Fisher metric of the model on the training data'''

    # layer_fim is list of blocks in FIM
    layer_fim = extract_blocks(fim)

    # Create a named tuple for better readability
    trans = namedtuple('transform', ['evals', 'evects'])
    eigen_pairs = [trans(*torch.lobpcg(block, k=num_stiff)) for block in layer_fim]

    reduced_fims = []
    for block_index, block in enumerate(layer_fim):
        # Block is the FIM for each layer
        transform = np.zeros_like(block)
        transform[:, num_stiff] = eigen_pairs[block_index].evects # ie stiff evects
        reduced_fims.append = transform @ np.diag(eigen_pairs[block_index].evals) @ transform.inv

    return reduced_fims

def _count_params(model):
    param_counts = []
    for param in model.parameters():
        param_counts.append(param.numel())
    return param_counts

def _find_layers(sloppy_rows, cum_param_count):
    '''
    Find the indices of layers of the model given cumulative parameter count
    list that corresponds to the largest elements smaller than or equal to 
    the values in `sloppy_rows`.

    Args:
        sloppy_rows (numpy.ndarray): A 1D numpy array containing values to
        search for (diagonal parameters of FIM which are weak).

        cum_param_count (numpy.ndarray): A 1D numpy array of cumulative
        parameter counts. It represents the cumulative sum of parameters in 
        each layer.

    Returns:
        numpy.ndarray: A 1D numpy array of indices corresponding to layers in
        `cum_param_count` where the largest elements smaller than or equal to
        the values in `sloppy_rows` are located. If no such element is found,
        the index will be set to 0.
    '''
    # Find the index of the first element greater than or equal to the input number
    # indices = np.searchsorted(cum_param_count, sloppy_rows)

    # Subtract 1 to get the index of the largest element smaller than the input number
    # layer_indices = np.maximum(indices - 1, 0)
    layer_indices = (sloppy_rows >= cum_param_count[0]).int()

    return layer_indices

def get_sloppy_rows(fim, model, cutoff, diag=True, encoder_param_end=15680):
    '''
    Extract rows corresponding to 'sloppy' parameters from the Fisher
    Information Matrix (FIM), based on a specified cutoff value. 'Sloppy'
    parameters are those with diagonal elements in the FIM below or equal to
    the cutoff value.

    Args:
        fim (torch.Tensor): The Fisher Information Matrix (FIM) as a square
        tensor.

        cutoff (float): The cutoff value used to determine 'sloppy' parameters.

        encoder_param_end (int): Number of parameters in the encoder (don't 
                                 wish to prune decoder).

    Returns:
        numpy.ndarray: A 1D numpy array containing row indices of the 'sloppy'
        parameters based on diagonal elements in the FIM that are below or equal
        to the cutoff.
    '''

    # Get the size of the square array (no params)
    n = fim.shape[0]

    # We first need to separate the layers since we'll want to change the
    # weights when we remove sloppy parameters. As is the FIM is of all layers
    param_counts = _count_params(model)
    assert n == sum(param_counts), f'Parameter count differs from number of params in FIM'

    n = encoder_param_end

    # Note the FIM indices are over each layer. Find which layer each sloppy
    # parameter lives in
    cum_param_count = np.cumsum(param_counts)
    print(cum_param_count)

    # Get the diagonal elements and their corresponding row indices
    diagonal_indices = torch.arange(n)

    if not diag:
        diagonal_elements = fim.diagonal()[:encoder_param_end]
    else:
        diagonal_elements = fim[:encoder_param_end]


    # Sort the diagonal elements along with their indices
    sorted_indices = diagonal_indices[diagonal_elements.argsort()]
    sorted_diagonal = diagonal_elements[sorted_indices]

    # Filter out elements below the cutoff and get corresponding row indices
    sloppy_indices = sorted_indices[sorted_diagonal <= cutoff]
    sloppy_rows = sloppy_indices

    return sloppy_rows

def zero_sloppy_params(fim, sloppy_rows, model):
    '''
    Zero out the parameters corresponding to `sloppy` diagonal elements in the
    Fisher Information Metric (FIM).

    Args:
        fim (numpy.ndarray): The Fisher Information Matrix (FIM) as a square
        numpy array (given by NNGeometry)

        sloppy_rows (numpy.ndarray): A 1D numpy array containing values
        corresponding to the 'sloppy' parameter rows to be zeroed out in the
        model params.

        model (torch.nn.Module): Torch model with 'sloppy' parameters to be
        zeroed

    Returns:
        torch.nn.Module: The modified PyTorch model with the 'sloppy' parameter
        rows zeroed out in the appropriate layers' weight matrices.
    '''
    # We first need to separate the layers since we'll want to change the
    # weights when we remove sloppy parameters. As is the FIM is of all layers
    param_counts = _count_params(model)

    # Note the FIM indices are over each layer. Find which layer each sloppy
    # parameter lives in
    cum_param_count = np.cumsum(param_counts)

    # The linear layers are the odd layers (1,3) but NNGeometry begins ordering
    # each useful layer (ignores activations) so it sees them as 0, 1.

    # Since our layers are alternating with activations, we can use 2n+1
    nn_model_layers = _find_layers(sloppy_rows, cum_param_count) # What NNGeom. calls a layer 
    torch_model_layers = 2 * nn_model_layers + 1
    # torch_model_layers = 2 * nn_model_layers + 1

    param_layer_sloppy = zip(sloppy_rows, torch_model_layers)

    with torch.no_grad():
        # For each sloppy direction, find which layer and set param row to 0
        for sloppy_dir, torch_layer_index in param_layer_sloppy:
            # Need the translation due to fact NNGeometry only cares about
            # layers with parameters
            assert torch_layer_index % 2 != 0, 'Expect torch_layer_index to be odd since even are activations'
            # layer_rel_param = sloppy_dir - cum_param_count[torch_layer_index]
            if torch_layer_index == 1:
                layer_rel_param = sloppy_dir
            else:
                layer_rel_param = sloppy_dir - cum_param_count[int(0.5 * (torch_layer_index - 1))]

            layer_shape = model.encoder[torch_layer_index].weight.data.shape

            # Remove (zero) the sloppy parameter
            model.encoder[torch_layer_index].weight.data[layer_rel_param // layer_shape[1], layer_rel_param % layer_shape[1]] = 0

    print(f'Removed {len(sloppy_rows)} params')
    return model, len(sloppy_rows)

