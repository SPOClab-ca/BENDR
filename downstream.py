import torch
import tqdm
import argparse

import objgraph

import time
import utils
from result_tracking import ThinkerwiseResultTracker

from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import Thinker
from dn3.trainable.processes import StandardClassification

from dn3_ext import BENDRClassification, LinearHeadBENDR

# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
mne.set_log_level(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=utils.MODEL_CHOICES)
    parser.add_argument('--ds-config', default="configs/downstream.yml", help="The DN3 config file to use.")
    parser.add_argument('--metrics-config', default="configs/metrics.yml", help="Where the listings for config "
                                                                                "metrics are stored.")
    parser.add_argument('--subject-specific', action='store_true', help="Fine-tune on target subject alone.")
    parser.add_argument('--mdl', action='store_true', help="Fine-tune on target subject using all extra data.")
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    args = parser.parse_args()
    experiment = ExperimentConfig(args.ds_config)
    if args.results_filename:
        results = ThinkerwiseResultTracker()

    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'):
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config)
        for fold, (training, validation, test) in enumerate(tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))):

            tqdm.tqdm.write(torch.cuda.memory_summary())

            if args.model == utils.MODEL_CHOICES[0]:
                model = BENDRClassification.from_dataset(training, multi_gpu=args.multi_gpu)
            else:
                model = LinearHeadBENDR.from_dataset(training)

            if not args.random_init:
                model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
                                              freeze_encoder=args.freeze_encoder)
            process = StandardClassification(model, metrics=added_metrics)
            process.set_optimizer(torch.optim.Adam(process.parameters(), ds.lr, weight_decay=0.01))

            # Fit everything
            process.fit(training_dataset=training, validation_dataset=validation, warmup_frac=0.1,
                        retain_best=retain_best, pin_memory=False, **ds.train_params)

            if args.results_filename:
                if isinstance(test, Thinker):
                    results.add_results_thinker(process, ds_name, test)
                else:
                    results.add_results_all_thinkers(process, ds_name, test, Fold=fold+1)
                results.to_spreadsheet(args.results_filename)

            # explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            objgraph.show_backrefs(model, filename='sample-backref-graph.png')
            del model
            torch.cuda.synchronize()
            time.sleep(10)

        if args.results_filename:
            results.performance_summary(ds_name)
            results.to_spreadsheet(args.results_filename)
