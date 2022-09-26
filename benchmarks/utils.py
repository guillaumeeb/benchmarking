import datetime
import logging
import os
from contextlib import contextmanager
from time import sleep, time

import fsspec
import pandas as pd
from distributed import Client
from dask.utils import format_bytes
from fsspec.implementations.local import LocalFileSystem
from distributed.diagnostics.plugin import UploadDirectory

from . import __version__
from .datasets import timeseries
from .ops import (
    anomaly,
    climatology,
    deletefile,
    get_version,
    openfile,
    readfile,
    spatial_mean,
    temporal_mean,
    writefile,
)

logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)


here = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
results_dir = os.path.join(here, 'results')


class DiagnosticTimer:
    def __init__(self):
        self.diagnostics = []

    @contextmanager
    def time(self, time_name, **kwargs):
        tic = time()
        yield
        toc = time()
        kwargs[time_name] = toc - tic
        self.diagnostics.append(kwargs)

    def dataframe(self):

        return pd.DataFrame(self.diagnostics)





class Runner:
    def __init__(self, input_file):
        import yaml

        try:
            with open(input_file) as f:
                self.params = yaml.safe_load(f)
        except Exception as exc:
            raise exc
        self.operations = {}
        self.operations['computations'] = [spatial_mean, temporal_mean, climatology, anomaly]
        self.operations['readwrite'] = [writefile, openfile, readfile, deletefile]
        self.operations['write'] = [writefile]
        self.operations['read'] = [openfile, readfile]
        self.client = None

    def create_cluster(self, cluster_manager, processes=1, **cluster_kwargs):
        """ Creates a Dask cluster using dask_jobqueue or dask_gateway """
        logger.warning('Creating a Dask cluster')
        logger.warning(f'Cluster Manager: {cluster_manager}')
        logger.warning(f'Kwargs: {cluster_kwargs}')
        
        if cluster_manager in ('pbs', 'slurm'):

            from dask_jobqueue import PBSCluster, SLURMCluster

            job_schedulers = {'pbs': PBSCluster, 'slurm': SLURMCluster}

            # Note about OMP_NUM_THREADS=1, --threads 1:
            # These two lines are to ensure that each benchmark workers
            # only use one threads for benchmark.
            # in the job script one sees twice --nthreads,
            # but it get overwritten by --nthreads 1
            cluster = job_schedulers[cluster_manager](
                processes=proecesses,
                local_directory='$TMPDIR',
                interface='ib0',
                env_extra=['OMP_NUM_THREADS=1'],
                extra=['--nthreads 1'],
                **cluster_kwargs
            )

            logger.warning(
                '************************************\n'
                'Job script created by dask_jobqueue:\n'
                f'{cluster.job_script()}\n'
                '***************************************'
            )
        elif cluster_manager == 'gateway':
            if processes > 1:
                logger.warning(f'Processing kwarg of value {processes} will be ignored with dask-gateway clusters.')
            from dask_gateway import Gateway
            gateway = Gateway()
            cluster = gateway.new_cluster(**cluster_kwargs)
        else:
            raise ValueError(f"Unkown Cluster Manager: {cluster_manager}")

        self.client = Client(cluster)
        logger.warning(f'Dask cluster dashboard_link: {self.client.cluster.dashboard_link}')
        
        if cluster_manager == 'gateway':
            #We need to upload benchmarking Python files to use them on worker side
            logger.warning(f'Uploading directory {here}')
            plugin = UploadDirectory(here, restart=True, update_path=True)
            self.client.register_worker_plugin(plugin, nanny=True)

    def run(self):

        logger.warning('Reading configuration YAML config file')
        operation_choice = self.params['operation_choice']
        machine = self.params['machine']
        cluster_manager = self.params['cluster_manager']
        cluster_kwargs = self.params['cluster_kwargs']
        chunk_per_worker = self.params['chunk_per_worker']
        freq = self.params['freq']
        spil = self.params['spil']
        output_dir = self.params.get('output_dir', results_dir)
        now = datetime.datetime.now()
        output_dir = os.path.join(output_dir, f'{machine}/{str(now.date())}')
        os.makedirs(output_dir, exist_ok=True)
        parameters = self.params['parameters']
        num_workers = parameters['number_of_workers_per_nodes']
        num_threads = parameters.get('number_of_threads_per_workers', 1)
        num_nodes = parameters['number_of_nodes']
        chunking_schemes = parameters['chunking_scheme']
        io_formats = parameters['io_format']
        filesystems = parameters['filesystem']
        fixed_totalsize = parameters['fixed_totalsize']
        chsz = parameters['chunk_size']
        #TODO Dump the environment somewhere
        env_export_filename = f"{output_dir}/env_export_{now.strftime('%Y-%m-%d_%H-%M-%S')}.yml"
        for wpn in num_workers:
            self.create_cluster(
                cluster_manager=cluster_manager,
                processes=wpn,
                **cluster_kwargs
            )
            for num in num_nodes:
                self.client.cluster.scale(num * wpn)
                self.client.wait_for_workers(n_workers=num * wpn, timeout=1800)
                
                timer = DiagnosticTimer()
                logger.warning(
                    '#####################################################################\n'
                    f'Dask cluster:\n'
                    f'\t{self.client.cluster}\n'
                )
                now = datetime.datetime.now()
                csv_filename = f"{output_dir}/compute_study_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
                for chunk_size in chsz:

                    for io_format in io_formats:

                        for filesystem in filesystems:

                            if filesystem == 's3':
                                if (io_format == 'netcdf') & (
                                    operation_choice == 'readwrite' or operation_choice == 'write'
                                ):
                                    logger.warning(
                                        f'### Skipping NetCDF S3 {operation_choice} benchmarking ###\n'
                                    )
                                    continue
                                
                                profile = self.params['profile']
                                bucket = self.params['bucket']
                                endpoint_url = self.params['endpoint_url']
                                    
                                #We need to get access/secret keys from the Client notebook AWS credential files.
                                #In the cloud, we have no guarantee that workers will have those files, so we
                                #need to passe those keys directly to the S3 filesystem object.
                                import boto3
                                session = boto3.Session(profile_name='default')
                                fs = fsspec.filesystem(
                                    's3',
                                    anon=False,
                                    key=session.get_credentials().access_key, 
                                    secret=session.get_credentials().secret_key,
                                    client_kwargs={'endpoint_url': endpoint_url},
                                    skip_instance_cache=True,
                                    use_listings_cache=True,
                                )
                                root = f'{bucket}'
                            elif filesystem == 'posix':
                                fs = LocalFileSystem()
                                local_dir = self.params['local_dir']
                                root = local_dir
                                if not os.path.isdir(f'{root}'):
                                    os.makedirs(f'{root}')
                            for chunking_scheme in chunking_schemes:
                                logger.warning(
                                    f'Benchmark starting with: \n\tworker_per_node = {wpn},'
                                    f'\n\tnum_nodes = {num}, \n\tchunk_size = {chunk_size},'
                                    f'\n\tchunking_scheme = {chunking_scheme},'
                                    f'\n\tchunk per worker = {chunk_per_worker}'
                                    f'\n\tio_format = {io_format}'
                                    f'\n\tfilesystem = {filesystem}'
                                )
                                ds, chunks = timeseries(
                                    fixed_totalsize=fixed_totalsize,
                                    chunk_per_worker=chunk_per_worker,
                                    chunk_size=chunk_size,
                                    chunking_scheme=chunking_scheme,
                                    io_format=io_format,
                                    num_nodes=num,
                                    freq=freq,
                                    worker_per_node=wpn,
                                )
                                if (chunking_scheme == 'auto') & (io_format == 'netcdf'):
                                    logger.warning(
                                        '### NetCDF benchmarking cannot use auto chunking_scheme ###'
                                    )
                                    continue
                                dataset_size = format_bytes(ds.nbytes)
                                logger.warning(ds)
                                logger.warning(f'Dataset total size: {dataset_size}')

                                for op in self.operations[operation_choice]:
                                    logger.warning(f'Operation begin: {op}')
                                    with timer.time(
                                        'runtime',
                                        operation=op.__name__,
                                        fixed_totalsize=fixed_totalsize,
                                        chunk_size=chunk_size,
                                        chunk_per_worker=chunk_per_worker,
                                        dataset_size=dataset_size,
                                        worker_per_node=wpn,
                                        threads_per_worker=num_threads,
                                        num_nodes=num,
                                        chunking_scheme=chunking_scheme,
                                        io_format=io_format,
                                        filesystem=filesystem,
                                        root=root,
                                        machine=machine,
                                        spil=spil,
                                        version=__version__,
                                        **cluster_kwargs
                                    ):
                                        fname = f'{chunk_size}{chunking_scheme}{filesystem}{num}'
                                        if op.__name__ == 'writefile':
                                            filename = op(ds, fs, io_format, root, fname)
                                        elif op.__name__ == 'openfile':
                                            ds = op(fs, io_format, root, chunks, chunk_size)
                                        elif op.__name__ == 'deletefile':
                                            ds = op(fs, io_format, root, filename)
                                        else:
                                            op(ds)
                                    logger.warning(f'Operation done: {op}')
                        # kills ds, and every other dependent computation
                        logger.warning('Computation done')
                        self.client.cancel(ds)
                        temp_df = timer.dataframe()
                        deps_blob, deps_ver = get_version()
                        temp_df[deps_blob] = pd.DataFrame([deps_ver], index=temp_df.index)
                        temp_df.to_csv(csv_filename, index=False)

                logger.warning(f'Persisted benchmark result file: {csv_filename}')

            logger.warning(
                'Shutting down the client and cluster before changing number of workers per nodes'
            )
            self.client.cluster.close()
            logger.warning('Cluster shutdown finished')
            self.client.close()
            logger.warning('Client shutdown finished')

        logger.warning('=====> The End <=========')
