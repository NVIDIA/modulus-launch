import zarr
import cupy as cp
import numpy as np

def get_chunks_for_slice(zarr_array, slices):
    """
    Given a Zarr array and a tuple of slices, return the keys for all chunks that
    intersect with the slices.
    """

    # Calculate chunk indices for each dimension using numpy
    starts = np.array([s.start if s.start is not None else 0 for s in slices])
    stops = np.array(
        [
            s.stop if s.stop is not None else dim
            for s, dim in zip(slices, zarr_array.shape)
        ]
    )
    chunk_lens = np.array(zarr_array.chunks)

    start_idxs = (starts // chunk_lens).tolist()
    stop_idxs = (-(-stops // chunk_lens)).tolist()

    # Generate chunk ranges
    chunk_ranges = [
        np.arange(start, stop) for start, stop in zip(start_idxs, stop_idxs)
    ]

    # Use meshgrid to generate combinations of chunk indices
    grids = np.meshgrid(*chunk_ranges, indexing="ij")

    # Reshape and stack to get all combinations
    combined_indices = np.stack(grids, axis=-1).reshape(-1, len(chunk_ranges))

    # Generate keys
    keys = [
        str(tuple(row)).replace(", ", ".").replace("(", "").replace(")", "")
        for row in combined_indices
    ]

    # Get chunks from store
    chunks = [value for value in zarr_array.chunk_store.getitems(keys).values()]

    return chunks
