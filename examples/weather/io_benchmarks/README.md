# ERA5 IO Benchmarks

This directory contains IO benchmarks for loading ERA5 datasets given different file formats, chunking and compression algorithms. The primary focus is to test GPU enabled compression on loading Zarr datasets.

# How to Use
1. Make sure you have the CDS API key setup following [these instructions](https://cds.climate.copernicus.eu/api-how-to).
2. Fetch the ERA5 data needed for the benchmarks by running, `python fetch_era5.py`. This will download the data in the directory specified in the config file. Currently the configs are set up to download a small ~10GB chuck of data relevant to training [FourCastNet](https://arxiv.org/abs/2202.11214).
3. Run the benchmarks by running, `python benchmark.py`. This will run the benchmarks for all the configurations specified in the config file. These tests will show, compression ratio, decompression time, and loading time for each configuration. All results will be saved as bar plots. 
4. If performing benchmarks on multi-GPU devices (e.g. DGX machine) you can use mpirun to run the benchmark in parallel. For example, `mpirun -np 8 --allow-run-as-root --rankfile rankfile ./set_device.sh python benchmark.py` will run these benchmarks on an 8 GPU machine. The `rankfile` and `set_device.sh` are provided as reference.

# Note
- Some machines may cache the data in memory, which can cause variable and misleading results. To avoid this, you can run the following between benchmarks, `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`.
- These IO tests depend on many factors such as the CPU, GPU, RAM and disk speed.

# Configuration File

The config files contain several configurations you can modify,

- `base_path`: Path where ERA5 data will be downloaded.
- `dt`: Time resolution in hours.
- `nr_months': Number of months to download.
- `decompression_slice`: Slice to be used for decompression benchmark.
- `throughput_slice`: Slice to be used for throughput benchmark.
- `experiments`: List of experiments to run.
-- `filetype`: File format to use. `zarr` or `hdf5`.
-- `device`: Compression device to use. `cpu` or `gpu`.
-- `compression_algorithm`: Compression algorithm to use. Check `experiment.py` for available algorithms.
-- `zarr_loading`: Either use Kvikio `GDSStore` or custom `MemMapStore` for loading Zarr files.
-- `chunking`: Chunking to use for Zarr files.
- `variables`: List of variables to download.

# Results

Complete benchmarks are still underway however here are some preliminary results performed on a 80GB A100 DGX machine. We see that using Gdeflate we are able to achieve an ~1.3 compression ratio. This directly translates to a throughput speed up when loading on all 8 GPUs.

![Compression Ratio](./results/compression_ratios.png)
![Throughput](./results/throughput_times.png)

