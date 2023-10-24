from mup import set_base_shapes
from mup.coord_check import get_coord_data, plot_coord_data
import os
import numpy as np


def coord_check(
    model_fn,
    mup,
    lr,
    train_loader,
    nsteps,
    nseeds,
    plotdir="",
    legend=False,
    in_dim=28**2,
    out_dim=10,
    nlayers=3,
):
    def gen(w, standparam=False):
        def f():
            model = model_fn(
                widths=[in_dim] + [w] * (nlayers - 2) + [out_dim], use_mup=mup
            )
            return model

        return f

    widths = 2 ** np.arange(7, 14)
    models = {w: gen(w, standparam=not mup) for w in widths}

    df = get_coord_data(
        models,
        train_loader,
        mup=mup,
        lr=lr,
        optimizer="sgd",
        flatten_input=True,
        nseeds=nseeds,
        nsteps=nsteps,
        lossfn="mse",
    )

    prm = "μP" if mup else "SP"
    return plot_coord_data(
        df,
        legend=legend,
        save_to=os.path.join(plotdir, f"{prm.lower()}_mlp_sgd_coord.png"),
        suptitle=f"{prm} MLP SGD lr={lr} nseeds={nseeds}",
        face_color="xkcd:light grey" if not mup else None,
    )
