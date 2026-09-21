"""Draw six validation-curve panels from an aggregate CSV table.

The input is deliberately a small, portable summary rather than a training
log or a dataset.  The script never looks outside the paths supplied on the
command line (the checked-in CSV is the default).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROUNDS = [1] + list(range(15, 301, 15))
METHODS = ['fedavg', 'fedpac', 'fedsa', 'cwfedavg', 'cwt', 'fedseq', 'fedtriad']
LABELS = ['FedAvg', 'FedPAC', 'FedSA', 'cwFedAvg', 'CWT', 'FedSeq', 'FedTriad']
COLORS = ['#666666', '#0072B2', '#009E73', '#E69F00', '#56B4E9', '#9368AC', '#D62728']
MARKERS = ['o', '^', 's', 'D', 'v', 'P', '*']
DATASETS = [('bloodmnist', 'BloodMNIST'), ('organamnist', 'OrganAMNIST'),
            ('pathmnist', 'PathMNIST')]


def load_summary(path):
    required = {'dataset', 'alpha', 'method', 'round', 'mean_percent',
                'sample_sd_percent', 'n_seeds'}
    data = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            missing = required - set(row)
            if missing:
                raise ValueError('Missing summary columns: ' + ', '.join(sorted(missing)))
            key = (row['dataset'], float(row['alpha']), row['method'], int(row['round']))
            if key in data:
                raise ValueError('Duplicate summary row: ' + repr(key))
            if key[2] not in METHODS or key[3] not in ROUNDS:
                continue
            if int(row['n_seeds']) != 3:
                raise ValueError('Expected three-seed rows: ' + repr(key))
            mean = float(row['mean_percent'])
            sd = float(row['sample_sd_percent'])
            if not (0 <= mean <= 100 and 0 <= sd <= 100):
                raise ValueError('Metric outside 0..100: ' + repr(key))
            data[key] = (mean, sd)
    expected = {(dataset, alpha, method, round_index)
                for alpha in (.5, .1) for dataset, _ in DATASETS
                for method in METHODS for round_index in ROUNDS}
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise ValueError('Summary coverage mismatch; missing=%s extra=%s' %
                         (missing[:3], extra[:3]))
    return data


def draw(data, stem, formats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.linewidth': .6, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'savefig.facecolor': 'white',
        'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 3, figsize=(5.15, 4.5))
    fig.subplots_adjust(left=.115, right=.955, bottom=.25, top=.90,
                        wspace=.38, hspace=.60)
    for i, alpha in enumerate([.5, .1]):
        for j, (dataset, title) in enumerate(DATASETS):
            ax = axes[i, j]
            for method, label, color, marker in zip(METHODS, LABELS, COLORS, MARKERS):
                values = [data[(dataset, alpha, method, round_index)][0]
                          for round_index in ROUNDS]
                ours = method == 'fedtriad'
                ax.plot(ROUNDS, values, color=color, label=label, marker=marker,
                        linewidth=1.05 if ours else .65,
                        markersize=3.8 if ours else 2.25, markeredgewidth=.35,
                        markeredgecolor=color,
                        markerfacecolor=color if ours else 'white',
                        zorder=10 if ours else 3, clip_on=False)
            ax.set_xlim(0, 300)
            ax.set_xticks([0, 60, 120, 180, 240, 300])
            ax.set_ylim(0, 100 if i == 0 else 60)
            ax.set_yticks(range(0, 101, 20) if i == 0 else range(0, 61, 10))
            ax.grid(True, linestyle='--', linewidth=.4, color='#C9C9C9', zorder=0)
            ax.tick_params(direction='out', length=2.2, width=.5, pad=2)
            ax.set_title(f'({chr(97 + i * 3 + j)}) {title}\n'
                         rf'$\alpha = {alpha}$', fontsize=8, pad=5)
    fig.supylabel('Client validation Macro-F1 (%)', x=.015, y=.60, fontsize=9)
    fig.supxlabel('Training rounds', x=.55, y=.15, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center',
        bbox_to_anchor=(.55, .015), ncol=4, frameon=False, fontsize=8,
        handlelength=1.7, columnspacing=1.0, handletextpad=.45,
        labelspacing=.6)
    legend.get_texts()[-1].set_fontweight('bold')
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    formats = list(dict.fromkeys(formats))
    pdftops = shutil.which('pdftops')
    if 'eps' in formats and pdftops is None:
        raise RuntimeError(
            'EPS output requires the Poppler pdftops executable; '
            'omit eps or install/configure pdftops.'
        )
    if 'pdf' in formats or 'eps' in formats:
        pdf_path = stem.with_suffix('.pdf')
        fig.savefig(pdf_path, dpi=220)
    for extension in formats:
        if extension in ('pdf', 'eps'):
            continue
        fig.savefig(stem.with_suffix('.' + extension), dpi=220)
    if 'eps' in formats:
        subprocess.run([pdftops, '-eps', str(pdf_path),
                        str(stem.with_suffix('.eps'))], check=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot six FedTriad validation-curve panels')
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).with_name('source_data_mean_sd.csv'),
                        help='Aggregate CSV with mean_percent and sample_sd_percent')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('FedTriad_validation_curves_6panels'),
                        help='Output filename stem (without extension)')
    parser.add_argument('--formats', nargs='+', default=['svg', 'pdf', 'png'],
                        choices=['svg', 'pdf', 'png', 'eps'])
    args = parser.parse_args()
    data = load_summary(args.input)
    draw(data, args.output, args.formats)
    manifest = {
        'panels': 6, 'methods': LABELS, 'rounds': ROUNDS, 'seeds': [0, 1, 2],
        'metric': 'Equal-client-mean validation Macro-F1, percent',
        'aggregation': 'Mean over 3 seeds; sample SD retained in input CSV.',
        'smoothing': False, 'round_zero_fabricated': False,
        'input': str(Path(args.input).resolve()),
        'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'mean_points': len(data),
    }
    manifest_path = Path(args.output).with_name(Path(args.output).name + '_manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(Path(args.output)), 'points': len(data),
                      'series': 42}, ensure_ascii=False))


if __name__ == '__main__':
    main()
