import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


PRETTY_NAMES = {
    'mmidb': 'MMI',
    'bci_iv_2a': 'BCIC',
    'ern': 'ERN',
    'erpbci': 'P300',
    'sleep-edf': 'SSC'
}

METRICS_DB = {
    'MMI': 'Accuracy',
    'BCIC': 'Accuracy',
    'ERN': 'auroc',
    'P300': 'auroc',
    'SSC': 'bac'
}

CHANCE_DB = {
    'MMI': 0.5,
    'BCIC': 0.25,
    'ERN': 0.5,
    'P300': 0.5,
    'SSC': 0.2
}

LEGEND_ORDER = [
    "Full",
    "Linear",
    "Full Random Init",
    "Full Frozen Encoder",
    "Linear Random Init",
    "Linear Frozen Encoder"
]


def downstream_plot_performance(df):
    print("Plotting...")
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    xtitle = 'Normalized Performance Metric'

    df = pd.melt(df, ['Model', 'Dataset'], value_vars='metric', value_name=xtitle)
    sns.stripplot(x=xtitle, y="Dataset", hue="Model", data=df, dodge=True, alpha=.25, zorder=1, size=2)
    sns.pointplot(x=xtitle, y="Dataset", hue="Model", data=df, dodge=0.65, join=False,
                  palette="pastel", errwidth=1, markers="d", scale=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[labels.index(l) + len(LEGEND_ORDER)] for i, l in enumerate(LEGEND_ORDER)],
              ["({}.) {}".format(i+1, l) for i, l in enumerate(LEGEND_ORDER)], title="Models",
              handletextpad=0, columnspacing=1, fontsize='x-small', loc="best", ncol=1, frameon=True)
    plt.title("Performance Metrics as Multiple of Chance-level By Dataset and Model")
    plt.show()
    print("Done.")


def sequence_plot_performance(df, n_boot=None):
    print("Plotting...")
    # sns.boxplot(x='Dataset', y='Accuracy', data=df, palette='pastel')
    sns.violinplot(x='Dataset', y='Accuracy', data=df, palette='pastel', inner='quartiles')
    # sns.stripplot(x="Dataset", y="Accuracy", data=df, color=".25")
    plt.title("Accuracy of Contrastive Task on Downstream Data")
    plt.show()
    print("Done.")


def regression_plot(df):
    print("Plotting...")
    xtitle = "Sequence Length (s)"
    # df = df.rename(dict(sequence_length=xtitle))
    df.loc[:, 'sequence_length'] /= 256
    # df.loc[:, 'Accuracy'] /= df.loc[:, 'Mask_pct']
    sns.lineplot(data=df, x='sequence_length', y="Accuracy", hue="Dataset", palette='pastel')
    plt.xscale('log')
    plt.xticks([20, 30, 40, 60])
    plt.xlabel(xtitle)
    plt.title("Contrastive Task vs. Sequence Length")
    plt.show()
    print("Done.")


def xlsx_to_df(spreadsheet):
    df = pd.concat(pd.read_excel(spreadsheet, sheet_name=None, engine='openpyxl').values(), ignore_index=True)
    model_name = Path(spreadsheet).stem.replace('_', ' ').title()
    model_name = model_name.replace('Bendr', 'BENDR')
    df['Model'] = [model_name] * len(df)
    return df.replace(PRETTY_NAMES)


def compile_performances_from_directory(directory):
    directory = Path(directory)
    dfs = list()
    print("Searching through:", directory)
    for spreadsheet in directory.glob('*.xlsx'):
        print("Reading:", spreadsheet)
        dfs.append(xlsx_to_df(spreadsheet))
    return pd.concat(dfs, ignore_index=True)


def downstream_plot(args):
    df = compile_performances_from_directory(args.directory)
    df['metric'] = [0] * len(df)
    for ds in METRICS_DB:
        df.loc[df['Dataset'] == ds, 'metric'] = (df[df['Dataset'] == ds][METRICS_DB[ds]] - CHANCE_DB[ds])/(1-CHANCE_DB[ds])
    downstream_plot_performance(df)


def sequence_likelihood_plot(args):
    sequence_plot_performance(xlsx_to_df(args.filename), n_boot=args.bootstrap)


def sequence_regression_plot(args):
    regression_plot(xlsx_to_df(args.filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summary analysis of BENDR models.")
    parser.add_argument('--metrics-config', default="configs/metrics.yml", help="Where the listings for config "
                                                                                "metrics are stored.")
    parser.add_argument('--bootstrap', default=1000, type=int, help="Number of bootstrap iterations to perform when "
                                                                    "estimating confidence intervals.")

    subparsers = parser.add_subparsers()

    downstream_parser = subparsers.add_parser('downstream', help='Plot all the downstream results from a directory.')
    downstream_parser.add_argument('directory', help="Directory containing '.xlsx' files with performance results.")
    downstream_parser.set_defaults(func=downstream_plot)

    sequence_parser = subparsers.add_parser('sequences', help='Plot the sequence likelihoods.')
    sequence_parser.add_argument('--filename', default='seq_results.xlsx', help='The name of the sequence results '
                                                                                'file.')
    sequence_parser.set_defaults(func=sequence_likelihood_plot)

    regression_parser = subparsers.add_parser('regression', help='Plot the sequence likelihoods.')
    regression_parser.add_argument('--filename', default='seq-regression.xlsx', help='The name of the sequence results '
                                                                                     'file.')
    regression_parser.set_defaults(func=sequence_regression_plot)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
