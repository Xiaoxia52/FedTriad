# Curve redraw

`source_data_mean_sd.csv` is the checked-in aggregate source table: one mean
and sample standard deviation over seeds 0/1/2 for each dataset, alpha,
method, and measured round. It contains no image, label, checkpoint, or raw
training log.

With `matplotlib` available, redraw the manuscript-size six-panel figure with:

```powershell
python .\figures\curves\plot_curves.py
```

Use `--input PATH` for another table and `--output PATH\figure_stem` to place
the outputs elsewhere. The plotter uses the checked visual specification:
2x3 panels, shared axis labels and legend, 8-point final typography, measured
rounds only (`1, 15, ..., 300`), and no smoothing or fabricated round zero.
When `eps` is requested, the script requires Poppler's `pdftops` and converts
the generated PDF to EPS; it refuses to emit a direct matplotlib EPS that may
lose Arial text.
