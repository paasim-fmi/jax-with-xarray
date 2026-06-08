import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import subprocess
import xarray as xr
import jax.numpy as jnp
import xarray_jax
from jax.scipy.signal import convolve2d


def to_xr(arr: xr.DataArray, shape: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        arr,
        coords=shape.coords,
        dims=shape.dims,
        name=shape.name,
        attrs=shape.attrs,
    )


def fill_na(y: xr.DataArray) -> xr.DataArray:
    kernel = jnp.ones((5, 5)) / 25
    f = xarray_jax.vmap(lambda v: convolve2d(v.data, kernel, mode="same"), dim="time")
    y_avg = f(y.fillna(y.mean()))
    return y.fillna(y_avg)


def fill_na_all(arr: xr.DataArray) -> xr.DataArray:
    return arr.fillna(arr.mean(dim=("time", "lat", "lon")))


def to_file(fig_name: str) -> str:
    return f"imgs/{fig_name}.png"


def plot(y, fig_name: str, time: int = 0, show=True):
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    y.isel(time=time).plot(
        ax=ax,
        x="lon",
        y="lat",
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        cbar_kwargs={"label": "Visibility"},
    )

    # ax.set_title(fig_name)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.7)
    ax.add_feature(cfeature.LAKES, alpha=0.3)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)
    plt.savefig(to_file(fig_name), dpi=200, bbox_inches="tight")
    if show:
        show_fig(fig_name)


def show_fig(fig_name: str):
    file_name = to_file(fig_name)
    _ = subprocess.run(["chafa", file_name])
