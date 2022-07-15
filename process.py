#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import json
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from util import hash_args
from scipy.interpolate import interp1d
from sklearn.metrics import auc
from util import get_roc

import numpy as np
import hydra
from hydra.utils import to_absolute_path

import logging
logger = logging.getLogger(__name__)


def get_fpr_tpr(fpr, tpr, thresholds, threshold0):
    fpr_tpr = interp1d(thresholds, np.concatenate((fpr[:, None], tpr[:, None]), axis=-1),
                       axis=0, kind='next')(threshold0)
    return fpr_tpr[0], fpr_tpr[1]


def process(inside, outside, threshold_f):
    results = {}
    roc = get_roc(inside=inside, outside=outside)
    fpr_f, tpr_f = get_fpr_tpr(*roc, threshold_f)
    accuracy_f = (1.0 - fpr_f + tpr_f) / 2.0
    fnr_f = 1.0 - tpr_f
    results['fpr_f'] = fpr_f
    results['tpr_f'] = tpr_f
    results['fnr_f'] = fnr_f
    results['accuracy_f'] = accuracy_f
    results['threshold_f'] = threshold_f

    results['min_inside'] = np.array(inside).min()
    results['max_outside'] = np.array(outside).max()

    fpr, tpr, thresholds = roc
    i_best = np.argmax(tpr - fpr)
    threshold_b = (thresholds[i_best] + thresholds[i_best+1]) / 2.0
    fpr_b, tpr_b = get_fpr_tpr(*roc, threshold_b)
    fnr_b = 1.0 - tpr_b
    accuracy_b = (1.0 - fpr_b + tpr_b) / 2.0
    results['fpr_b'] = fpr_b
    results['tpr_b'] = tpr_b
    results['fnr_b'] = fnr_b
    results['accuracy_b'] = accuracy_b
    results['threshold_b'] = threshold_b

    roc_auc = auc(fpr, tpr)
    results['roc_auc'] = roc_auc

    return results


@hydra.main(config_path='conf', config_name='process')
def main(args: DictConfig):
    output_directory = Path(to_absolute_path(args.runtime.output_dir))
    output_file = output_directory / 'results.json'

    results = {}
    for infile in output_directory.glob('*.json'):
        with open(infile, 'r') as fp:
            r0 = json.load(fp)
        if 'inside' not in r0:
            continue
        h = hash_args(OmegaConf.create(r0['param']))

        # check lengths
        if len(r0['inside']) != r0['param']['runs']:
            logger.warning(f"ERROR at {r0['param']}:\n  Got {len(r0['inside'])} inside samples, expected {r0['param']['runs']}.")
            continue
        elif len(r0['outside']) != r0['param']['runs']:
            logger.warning(f"ERROR at {r0['param']}:\n  Got {len(r0['outside'])} outside samples, expected {r0['param']['runs']}.")
            continue

        output = process(r0['inside'], r0['outside'], args.eval.threshold)
        output['param'] = r0['param']
        results[h] = output

    with output_file.open('w') as fp:
        json.dump(results, fp, indent=2)

    return output_file

if __name__ == '__main__':
    main()
