import os
import numpy as np
import torch
import mup
from mup.coord_check import get_coord_data, plot_coord_data
from typing import Iterable
import tqdm
import pandas as pd
from .Fire import Fire
from .models import get_optimizer
from matplotlib import pyplot as plt


def coord_check(
    model_fn,
    mup,
    lr,
    train_loader,
    nsteps,
    nseeds,
    plotdir="",
    legend=False,
    lossfn="cross_entropy",
):
    """make coord_check plot for a model. takes model callable which returns a model with the given width."""

    def gen(w):
        def f():
            model = model_fn(w)
            return model

        return f

    widths = 2 ** np.arange(7, 14)
    models = {w: gen(w) for w in widths}

    df = get_coord_data(
        models,
        train_loader,
        mup=mup,
        lr=lr,
        optimizer="sgd",
        flatten_input=True,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn=lossfn,
    )

    prm = "μP" if mup else "SP"
    return plot_coord_data(
        df,
        legend=legend,
        save_to=os.path.join(plotdir, f"{prm.lower()}_mlp_sgd_coord.png"),
        suptitle=f"{prm} MLP SGD lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
    )


def make_delta_args(fixed_args, scale_args):
    """make delta model by adding one to scale args, ie. the width-like parameters of the base model."""
    delta_args = fixed_args.copy()
    for k, v in scale_args.items():
        if isinstance(v, Iterable):
            orginal_type = type(v)
            delta_args[k] = orginal_type([x + 1 for x in v])
        else:
            delta_args[k] = v + 1
    return delta_args


def make_mup(
    model_fn,
    readout_fn,
    fixed_args,
    scale_args,
    savefile=None,
    reinit=True,
):
    """take model init function and return a mup model.
    This method expects the model to have readout layer(s) which will be replaced with MuReadout.
    Scale args are going to be used as default for the base model.
    readout_fn is a function that replaces the readout layer(s) with MuReadout and returns them.
    """
    ######### Setup Shapes #########
    base_args = fixed_args.copy()
    base_args.update(scale_args)
    base_model = model_fn(**base_args)

    delta_args = make_delta_args(fixed_args, scale_args)
    delta_model = model_fn(**delta_args)

    model = model_fn(**fixed_args)
    readouts = readout_fn(model)
    mup.set_base_shapes(model, base_model, delta_model, savefile=savefile)

    ######### Re-init #########
    if reinit:
        for name, p in model.named_parameters():
            if "bias" in name or "readout" in name:
                mup.init.uniform_(p, 0, 0)
            else:
                mup.init.kaiming_uniform_(p, a=None)
                # mup.init.uniform_(p, -0.1, 0.1)
        for readout in readouts:
            readout.weight.data = torch.zeros_like(readout.weight.data)
            if readout.bias is not None:
                readout.bias.data = torch.zeros_like(readout.bias.data)
    return model



def make_histories(
    model_fn,
    trainloader,
    widths,
    lrs,
    seeds,
    valloader=None,
    savefile=None,
    lossfn="mse",
    epochs=500,
):
    """train models with different widths and learning rates and save history to df."""
    # check if notebook to make tqdm.notebook.trange instead of tqdm.trange
    try:
        from IPython import get_ipython
        get_ipython()
        from tqdm.notebook import trange
    except:
        from tqdm import trange
    pbar = trange(len(widths) * len(seeds) * len(lrs))

    histories = []
    for lr in lrs:
        for width in widths:
            for seed in seeds:
                row = {"lr": lr, "width": width, "seed": seed}
                pbar.update(1)
                torch.manual_seed(seed)
                model = model_fn(width)
                optimizer = get_optimizer(model, lr, use_mup=True)
                flame_model = Fire(model, optimizer, device="cuda", loss=lossfn)
                flame_model.fit(
                    trainloader, valloader=valloader, epochs=epochs, track_loss=10
                )
                for k, v in flame_model.history.items():
                    row[k] = v
                histories.append(row)
                desc = f"lr: {lr}, width: {width}, seed: {seed}"
                desc += ",".join(
                    [f" {k}:{v[-1]}" for k, v in flame_model.history.items()]
                )
                pbar.set_description(desc)
    pbar.close()

    # export to csv
    df = pd.DataFrame(histories)
    if savefile:
        if savefile.endswith(".csv"):
            savefile = savefile[:-4]
        df.to_csv(f"{savefile}.csv")
    return df


def make_transfer_plot(df_src, transfer_key='lr', savefile=None):
    """plot final loss as a function of hyperparameters with different widths"""
    if isinstance(df_src, str):
        df = pd.read_csv(df_src, index_col=0)
    else:
        df = df_src.copy()

    metric_keys = [k for k in df.columns if k not in ["lr", "width", "seed", "epoch", transfer_key]]
    # CLEANUP df
    # we neede to plot final loss (metric) for each width as a function of learning rate
    for k in metric_keys:
        eval_fn = eval if not isinstance(df[k][0], list) else lambda x: x
        # when saving to csv, the list of losses is converted to string so we need to eval it
        df[k] = df[k].apply(eval_fn).apply(lambda x: x[-1])

    # SETUP plot
    fig = plt.figure(figsize=(8 * len(metric_keys), 5))
    for i, column_name in enumerate(metric_keys):
        # make shaded area for max and min and center is mean
        max_ = df.groupby([transfer_key, "width"])[column_name].max().unstack()
        min_ = df.groupby([transfer_key, "width"])[column_name].min().unstack()
        mean_ = df.groupby([transfer_key, "width"])[column_name].mean().unstack()

        x_axis = mean_.index

        fig.add_subplot(1, len(metric_keys), i + 1)
        for width in mean_.columns:
            plt.fill_between(x_axis, min_[width], max_[width], alpha=0.1)
            plt.plot(x_axis, mean_[width], label=width)

        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(transfer_key)
        plt.ylabel(f"{column_name}")
    if savefile is not None:
        plt.savefig(savefile)

def _convert_str_to_list(df):
    for k in df.columns:
        if isinstance(df[k][0], str):
            df[k] = df[k].apply(eval)
    return df
    

def plot_history(df_src, transfer_key="lr", savefile=None):
    if isinstance(df_src, str):
        df = pd.read_csv(df_src, index_col=0)
    else:
        df = df_src
    df = _convert_str_to_list(df)
    lrs = df[transfer_key].unique() # could be lr could be something else
    widths = df.width.unique()

    metric_keys = [k for k in df.columns if k not in ["lr", "width", "seed", "epoch", transfer_key]]
    fig = plt.figure(figsize=(8 * len(metric_keys), 5 * len(lrs)))
    cols, rows = len(metric_keys), len(lrs)
    for row, lr in enumerate(lrs.round(8)):
        for col, metric in enumerate(metric_keys):
            fig.add_subplot(rows, cols, col + row * cols + 1)
            group = df.groupby([transfer_key, "width"])[metric]

            for c, width in enumerate(widths):
                losses = group.get_group((lr, width))
                losses = np.array(losses.tolist())
                mean_ = losses.mean(axis=0)
                max_ = losses.max(axis=0)
                min_ = losses.min(axis=0)
                x_axis = np.arange(len(mean_)) * 10
                plt.fill_between(x_axis, min_, max_, alpha=0.1)
                plt.plot(x_axis, mean_, label=width)

            plt.xlabel("epoch")
            plt.ylabel(metric)
            plt.yscale("log")
            plt.title(f"{transfer_key}={lr} - {metric}")
            plt.legend(title="width")
    if savefile is not None:
        # split last . to append details before extension
        savefile = savefile.split(".")
        savefile.insert(-1, f"lr_{lr}")
        savefile = ".".join(savefile)
        plt.savefig(savefile)
