#!/usr/bin/env python
# *-* encoding: utf-8 *-*
import os.path
import random

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from filelock import SoftFileLock, Timeout
from tqdm import tqdm
import uuid
from util import hash_args, DatasetLoader, get_sample_loader

from dishonest_server import main as server_main
from client import main as client_main
from client import DATASETS, MyClient, load_data
from random import choice

import json
from pathlib import Path
import logging
import threading as th
from queue import Queue

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name='main')
def main(args: DictConfig):
    output_directory = Path(to_absolute_path(args.runtime.output_dir))
    output_directory.mkdir(parents=True, exist_ok=True)
    output_filename = hash_args(args.param) + '.json'
    output_file = output_directory / output_filename

    dataset_dir = Path(to_absolute_path(args.runtime.dataset_dir))

    do_runs(args, output_file=output_file, dataset_dir=dataset_dir)


def do_runs(args: DictConfig, output_file: Path, dataset_dir: Path):
    port = random.randint(10000, 60000)
    dloader = DatasetLoader(load_path=str(dataset_dir), duplicates_path=args.runtime.duplicates)

    lockfile = str(output_file) + '.lock'
    try:
        with SoftFileLock(lockfile, timeout=10):
            if output_file.exists():
                with output_file.open(mode='r') as fp:
                    results = json.load(fp)
            else:
                results = {}
                results['param'] = OmegaConf.to_container(args.param, resolve=True)

            for inside in (True, False):
                inside_str = 'inside' if inside else 'outside'
                if inside_str not in results:
                    sub_results = []
                    results[inside_str] = sub_results
                else:
                    sub_results = results[inside_str]
                left = args.param.runs - len(sub_results)
                if left <= 0:
                    logger.info(f"\"{inside_str}\" already done.")
                    continue
                for _ in tqdm(range(left)):
                    score = run(args=args, dloader=dloader, port=port, inside=inside)
                    sub_results.append(score)
                    with output_file.open(mode='w') as fp:
                        json.dump(obj=results, fp=fp, indent=2)
    except Timeout:
        logger.error(f'Unable to lock {output_file}. Maybe delete stale {lockfile}.')


def run(args, dloader: DatasetLoader, port: int, inside: bool):
    q = Queue()

    dataset = {x.__name__: x for x in DATASETS}[args.dataset.name]
    trainset = dloader(dataset)

    trainloader = load_data(trainset=trainset,
                            batchsize=args.param.batchsize,
                            batches=args.param.batches)

    client = MyClient(trainloader=trainloader,
                      dataset=dataset,
                      optimizer=args.optimizer.name,
                      epochs=args.param.epochs)

    orig_state_dict = client.net.state_dict()

    if inside:
        index = choice(range(len(trainloader.dataset)))
        sample_dataset = trainloader.dataset
    else:
        indices = list(set(range(len(trainset))) - set(trainloader.dataset.indices))
        index = choice(indices)
        sample_dataset = trainset
    sample_loader = get_sample_loader(dataset=sample_dataset, index=index)
    srv = th.Thread(target=server_main,
                    daemon=False,
                    kwargs={'sample_loader': sample_loader,
                            'dataset': dataset,
                            'batchsize': args.param.batchsize,
                            'batches': args.param.batches,
                            'optimizer': args.optimizer.name,
                            'queue': q,
                            'epsilon': args.param.epsilon,
                            'top_j': args.param.top_j,
                            'port': port})

    cli = th.Thread(target=client_main,
                    daemon=False,
                    kwargs={'client': client,
                            'port': port})

    srv.start()
    cli.start()

    cli.join()
    srv.join()

    val = q.get()

    # # Debug output False Positives:
    # if not inside and val >= 0.1:
    #     output_directory = Path(to_absolute_path(args.runtime.output_dir))
    #     err_filename = f"{hash_args(args.param)}_{uuid.uuid4()!s}"
    #     with (output_directory / (err_filename + '.json')).open('w') as fp:
    #         json.dump({'index': index,
    #                    'indices': trainloader.dataset.indices,
    #                    'param': OmegaConf.to_container(args.param, resolve=True),
    #                    'val': val}, fp=fp, indent=1)
    #     torch.save(orig_state_dict, output_directory / (err_filename + '_orig.state'))
    #     torch.save(client.net.state_dict(), output_directory / (err_filename + '_final.state'))

    return val


if __name__ == '__main__':
    for topic in ("flower",):
        logging.getLogger('flower').setLevel(logging.WARNING)
    main()
