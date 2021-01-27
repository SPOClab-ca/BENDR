import tqdm
import argparse

import numpy as np

import utils
from dn3_ext import ConvEncoderBENDR, BENDRContextualizer, BendingCollegeWav2Vec
from result_tracking import ThinkerwiseResultTracker

from dn3.configuratron import ExperimentConfig
from dn3.transforms.instance import TemporalCrop

# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
mne.set_log_level(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('--pretraining-config', default='configs/pretraining.yml')
    parser.add_argument('--sequence-config', default="configs/sequence_evaluation.yml")
    parser.add_argument('--hidden-size', default=512, type=int, help="The hidden size of the encoder.")
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default='seq_results.xlsx', help='What to name the spreadsheet '
                                                                               'produced with all final results.')
    parser.add_argument('--min-sequence', default=None, type=int, help='Sequence regression starting point.')
    parser.add_argument('--num-sequence', default=None, type=int, help='Number of regression points.')
    return parser.parse_args()


def run(dataset, ds_name, args, pre_exp, seq_exp, result_tracker):
    encoder = ConvEncoderBENDR(20, encoder_h=args.hidden_size)
    tqdm.tqdm.write(encoder.description(pre_exp.global_sfreq, dataset.sequence_length))
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=pre_exp.bending_college_args.layer_drop)
    encoder.load(seq_exp.encoder_weights)
    contextualizer.load(seq_exp.context_weights)

    process = BendingCollegeWav2Vec(encoder, contextualizer, **pre_exp.bending_college_args)
    result_tracker.add_results_all_thinkers(process, ds_name, dataset, sequence_length=dataset.sequence_length)


if __name__ == '__main__':
    args = parse_args()
    pretrain_experiments = ExperimentConfig(args.pretraining_config)
    sequence_experiments = ExperimentConfig(args.sequence_config)
    results = ThinkerwiseResultTracker()

    for ds_name, ds in tqdm.tqdm(sequence_experiments.datasets.items(),
                                 total=len(sequence_experiments.datasets.items()), desc='Datasets'):
        dataset = utils.get_ds(ds_name, ds)

        if args.min_sequence is not None and args.num_sequence is not None:
            # Skip this dataset
            if ds_name == 'erpbci':
                continue
            logspace = list(reversed(np.logspace(np.log10(args.min_sequence),
                                                 np.log10(sequence_experiments.global_samples),
                                                 num=args.num_sequence).astype(int)))
            for seq_len in tqdm.tqdm(logspace, desc='Sequence length'):
                tqdm.tqdm.write("Length of sequences: {}".format(seq_len))
                dataset.add_transform(TemporalCrop(int(seq_len)))
                run(dataset, ds_name, args, pretrain_experiments, sequence_experiments, results)
        else:
            run(dataset, ds_name, args, pretrain_experiments, sequence_experiments, results)

        results.to_spreadsheet(args.results_filename)


