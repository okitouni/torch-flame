import os
import numpy as np
import torch
import mup
from mup.coord_check import get_coord_data, plot_coord_data
from typing import Iterable

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
    def gen(w):
        def f():
            model = model_fn(w=w)
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
    delta_args = fixed_args.copy()
    for k, v in scale_args.items():
        if isinstance(v, Iterable):
            orginal_type = type(v)
            delta_args[k] = orginal_type([x + 1 for x in v])
        else:
            delta_args[k] = v + 1
    return delta_args

def make_mup(model_fn, readout_fn, fixed_args, scale_args, savefile=None, ):
    """take model init function and return a mup model.
    This method expects the model to have readout layer(s) which will be replaced with MuReadout.
    Scale args are going to be used as default for the base model."""
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
