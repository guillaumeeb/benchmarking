# Benchmarking

Benchmarking &amp; Scaling Studies of the Pangeo Platform

- [Benchmarking](#benchmarking)
  - [Creating an Environment on an HPC Center](#creating-an-environment-on-an-hpc-center)
  - [Environment on a Kubernetes based system](#environment-on-a-kubernetes-based-system)
  - [Benchmark Configuration](#benchmark-configuration)
  - [Running the Benchmarks](#running-the-benchmarks)
  - [Benchmark Results](#benchmark-results)
  - [Visualization](#visualization)

## Creating an Environment on an HPC Center

To run the benchmarks on an HPC platform, it's recommended to create a dedicated conda environment by running:

```bash
conda env create -f ./binder/environment.yml
```

This will create a conda environment named `pangeo-bench` with all of the required packages.

You can activate the environment with:

```bash
conda activate pangeo-bench
```

and then run the post build script:

```bash
./binder/postBuild
```

## Environment on a Kubernetes based system

To run the benchmark on any Cloud platform using Kubernetes, it is recommanded to use [pangeo/pangeo-notebook Docker image](https://github.com/pangeo-data/pangeo-docker-images/tree/master/pangeo-notebook).

This package currently assumes a Dask Gateway cluster is available from the Kubernetes environment.

## Benchmark Configuration

The `benchmark-configs` directory contains YAML files that are used to run benchmarks on different machines. So far, HPC systems config have been provided for several clusters: Cheyenne from NCAR, HAL from CNES, Wrangler from TACC. It also contains configurations for CESNET Center based on a Kubernetes deployment over Openstack. There might be several configurations for each center.

In case you are interested in running the benchmarks on another system, you will need to create a new YAML file for your system with the right configurations. See the existing config files for reference.

## Running the Benchmarks

### from command line

To run the benchmarks, a command utility `pangeobench` is provided in this repository.
To use it to benchmark Pangeo computation, you need to specify subcommand `run` and the location of the benchmark configuration.

```bash
./pangeobench run benchmark-configs/cheyenne.pri2.yaml
```


To use it to benchmark Pangeo IO with weak scaling analysis, you need to specify subcommand `run` and the location of the benchmark configuration.


```bash
./pangeobench run benchmark-configs/cheyenne.readwrite.yaml
```

To use it to benchmark Pangeo IO with strong scaling analysis, you need the following three steps

First, create data files:
```bash
./pangeobench run benchmark-configs/cheyenne.write.yaml
```
Second, upload data files to S3 object store if you need to benchmark S3 object store:
```bash
./pangeobench upload --config_file benchmark-configs/cheyenne.write.yaml
```

Last, read data files:
```bash
./pangeobench run benchmark-configs/cheyenne.read.yaml
```

```bash
$ ./pangeobench --help
Usage: pangeobench [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  run     Run benchmarking
  upload  Upload benchmarking files from local directory to S3 object store
```

### from Jupyter notebook.

To run the benchmarks from jupyter notebook, install 'pangeo-bench' kernel to your jupyter notebook enviroment, then start run.ipynb notebook.  You will need to specify the configuration file as described above in your notebook.

To install your 'pangeo-bench' kernel to your jupyter notebook enviroment you'll need to connect a terminal of your HPC enviroment and run following command.

```conda env create -f pangeo-bench.yml
source activate pangeo-bench
ipython kernel install --user --name pangeo-bench
```

Before starting your jupyternotebook, you can verify that if your kernel is well installed or not by follwing command

```
jupyter kernelspec list
```



## Benchmark Results

Benchmark results are persisted in the `results` directory by default. The exact location of the benchmark results depends on the machine name (specified in the config file) and the date on which the benchmarks were run. For instance, if the benchmarks were run on Cheyenne supercomputer on 2019-09-07, the results would be saved in: `results/cheyenne/2019-09-07/` directory. The file name follows this template: `compute_study_YYYY-MM-DD_HH-MM-SS.csv`

## Visualization

Visualisation can be done using jupyter notebooks placed in analysis directories.
