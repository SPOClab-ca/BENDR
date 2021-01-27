import torch
import yaml

from dn3.metrics.base import balanced_accuracy, auroc
from dn3.transforms.instance import To1020

from dn3_ext import LoaderERPBCI, LinearHeadBENDR, BENDRClassification


CUSTOM_LOADERS = dict(
    erpbci=LoaderERPBCI,
)

EXTRA_METRICS = dict(bac=balanced_accuracy,
                     auroc=auroc)

MODEL_CHOICES = ['BENDR', 'linear']


def make_model(args, experiment, dataset):
    if args.model == MODEL_CHOICES[0]:
        model = BENDRClassification.from_dataset(dataset)
    else:
        model = LinearHeadBENDR.from_dataset(dataset)

    if not args.random_init:
        model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
                                      freeze_encoder=args.freeze_encoder)

    return model


def get_ds_added_metrics(ds_name, metrics_config):
    """
    Given the name of a dataset, and name of metrics config file, returns all additional metrics needed,
    the metric to retain the best validation instance of and the chance-level threshold of this metric.
    """
    metrics = dict()
    retain_best = 'Accuracy'
    chance_level = 0.5

    with open(metrics_config, 'r') as f:
        conf = yaml.safe_load(f)
        if ds_name in conf:
            metrics = conf[ds_name]
            if isinstance(metrics[0], dict):
                metrics[0], chance_level = list(metrics[0].items())[0]
            retain_best = metrics[0]

    return {m: EXTRA_METRICS[m] for m in metrics if m != 'Accuracy'}, retain_best, chance_level


def get_ds(name, ds):
    if name in CUSTOM_LOADERS:
        ds.add_custom_raw_loader(CUSTOM_LOADERS[name]())
    dataset = ds.auto_construct_dataset()
    dataset.add_transform(To1020())
    return dataset


def get_lmoso_iterator(name, ds):
    dataset = get_ds(name, ds)
    specific_test = ds.test_subjects if hasattr(ds, 'test_subjects') else None
    iterator = dataset.lmso(ds.folds, test_splits=specific_test) \
        if hasattr(ds, 'folds') else dataset.loso(test_person_id=specific_test)
    return iterator


# See - https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert (isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    gc.collect()
                    referrers = gc.get_referrers(obj)
                    for referrer in referrers:
                        print(namestr(referrer, globals()))
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)
