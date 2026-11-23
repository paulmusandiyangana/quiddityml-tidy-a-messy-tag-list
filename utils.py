"""Data loaders and plot helpers for this project (from QuiddityML)."""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

raw_tags = "  NLP, vision ,Python,nlp , Vision,data ,python, NLP,robotics , Data,VISION, ethics,speech, Ethics\n"



# ---- Plotting outside QuiddityML: matplotlib versions of the plot helpers above ----
import matplotlib.pyplot as plt

_COLORS = ['#e8b44a', '#6ea8fe', '#ec7c6c', '#b49cff', '#ff8fc4', '#555555']
_LABEL_COLORS = ['#6ea8fe', '#e8b44a', '#ec7c6c', '#b49cff']
# Blue where a model says 0, amber where it says 1, as in the app.
from matplotlib.colors import LinearSegmentedColormap as _LSC
_BLUE_AMBER = _LSC.from_list('blue_amber', ['#6ea8fe', '#f4f1ea', '#e8b44a'])


def _np(t):
    """A tensor or list as a NumPy array."""
    if hasattr(t, 'detach'):
        t = t.detach()
    if hasattr(t, 'numpy'):
        return t.numpy()
    return np.asarray(t)


def _finish(ax, title):
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_points(X, y, title=''):
    X, y = _np(X), _np(y).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 4))
    for k in sorted(set(int(v) for v in y)):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=14, color=_LABEL_COLORS[k % 4], label=f'label {k}')
    ax.legend()
    _finish(ax, title)


def plot_boundary(model, X, y, title='', features=None, n=200):
    Xn, yn = _np(X), _np(y).reshape(-1)
    x0, x1 = Xn[:, 0].min() - 0.3, Xn[:, 0].max() + 0.3
    y0, y1 = Xn[:, 1].min() - 0.3, Xn[:, 1].max() + 0.3
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    G = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    if features is not None:
        G = features(G)
    with torch.no_grad():
        p = _np(model(G)).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contourf(gx, gy, p, levels=20, cmap=_BLUE_AMBER, alpha=0.55)
    for k in sorted(set(int(v) for v in yn)):
        m = yn == k
        ax.scatter(Xn[m, 0], Xn[m, 1], s=14, color=_LABEL_COLORS[k % 4], edgecolor='k', linewidth=0.3, label=f'label {k}')
    ax.legend()
    _finish(ax, title)


def plot_surface(loss_at, x_range, y_range, paths=None, minimum=None, title=''):
    (a0, a1), (b0, b1) = x_range, y_range
    gx, gy = np.meshgrid(np.linspace(a0, a1, 120), np.linspace(b0, b1, 120))
    z = np.vectorize(lambda u, v: float(loss_at(u, v)))(gx, gy)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contourf(gx, gy, np.log1p(z), levels=30, cmap='plasma')
    for k, (label, path) in enumerate(paths or []):
        P = _np(path)
        ax.plot(P[:, 0], P[:, 1], '.-', color=['white', '#7fb8ff', '#ff8fc4', '#c9b6ff'][k % 4], label=label)
    if minimum is not None:
        ax.plot([minimum[0]], [minimum[1]], '*', color='white', markersize=14)
    ax.set_xlim(a0, a1)
    ax.set_ylim(b0, b1)
    if paths:
        ax.legend()
    _finish(ax, title)


def plot_curves(curves, title='', ylabel='loss', log=True):
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, (label, values) in enumerate(curves):
        ax.plot(_np(values), color=_COLORS[k % 6], label=label)
    if log:
        ax.set_yscale('log')
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    ax.legend()
    _finish(ax, title)


def plot_lines(series, title='', xlabel='step', ylabel='', log=False, logx=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, item in enumerate(series):
        if len(item) == 2:
            label, ys = item
            ax.plot(_np(ys), color=_COLORS[k % 6], label=label)
        else:
            label, xs, ys = item
            ys = _np(ys)
            ax.plot(_np(list(xs))[:len(ys)], ys, color=_COLORS[k % 6], label=label)
    if log:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _finish(ax, title)


def plot_fit(X, y, models=None, title=''):
    Xn, yn = _np(X).reshape(-1), _np(y).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(Xn, yn, s=12, color='#9a96b8')
    xs = np.linspace(Xn.min(), Xn.max(), 50)
    for k, (label, model) in enumerate(models or []):
        a, b = slope_and_intercept(model)
        ax.plot(xs, a * xs + b, color=_COLORS[k % 6], label=label)
    if models:
        ax.legend()
    _finish(ax, title)


def plot_bars(categories, series, title='', ylabel='', log=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    n = max(1, len(series))
    width = 0.8 / n
    for k, (label, values) in enumerate(series):
        xs = np.arange(len(categories)) + (k - (n - 1) / 2) * width
        ax.bar(xs, [float(v) for v in values], width, color=_COLORS[k % 6], label=label)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels([str(c) for c in categories])
    if log:
        ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    if n > 1:
        ax.legend()
    _finish(ax, title)


def plot_images(images, captions=None, title='', bad=None):
    imgs = _np(images)
    count = imgs.shape[0]
    side = int(round(imgs[0].size ** 0.5))
    cols = min(count, 8)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.2 * cols, 1.4 * rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis('off')
        if k < count:
            ax.imshow(imgs[k].reshape(side, side), cmap='magma')
            if captions:
                ax.set_title(str(captions[k]), fontsize=8, color='#d9534f' if bad and bad[k] else 'black')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
