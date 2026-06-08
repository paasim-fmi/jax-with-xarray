import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import xarray as xr
import xarray_jax # <----- woah
from figs import plot, to_xr, fill_na

# we can do almost any numpy function, just remember to export it from
# jax.numpy
def sq(x):
    return jnp.sqrt(x)

# because jax is an autodiff, we can also do gradients, hessians etc.
d_sq = jax.grad(sq)

# f(x) = x^0.5, f'(x) = 0.5 * 1/sqrt(x)

sq(1.0)   # f(1)  = 1
d_sq(1.0) # f'(1) = 0.5

# Next lets do some xarray-stuff

# As an example we just use a random meps forecast, e.g.
# (you need to adjust the dates)
# https://opendata.fmi.fi/download?producer=harmonie_scandinavia_surface&param=Pressure,GeopHeight,Temperature,DewPoint,Humidity,WindDirection,WindSpeedMS,WindUMS,WindVMS,PrecipitationAmount,CAPE,TotalCloudCover,LowCloudCover,MediumCloudCover,HighCloudCover,RadiationGlobal,Visibility,WindGust,RadiationGlobalAccumulation,RadiationNetSurfaceLWAccumulation,RadiationNetSurfaceSWAccumulation,RadiationSWAccumulation&bbox=18,55,35,74&origintime=2026-06-08T06:00:00Z&starttime=2026-06-08T06:00:00Z&endtime=2026-06-11T00:00:00Z&format=netcdf&projection=EPSG:4326&levels=0&timestep=60
ds = xr.open_dataset("data/harmonie.nc").drop_dims("time_h")

# Visibility in km, 349 null values, fix somehow
y = fill_na(ds["visibility_in_air_407"] / 1000)
plot(y, "visibility")



# jax also contains more exotic functions from scipy; see how even with xarray
# this "just" works once xarray_jax is imported. you just use .data to access the array
def conv2d(kernel: jax.Array, vis: xr.DataArray) -> xr.DataArray:
    # map over time, ie. for each timestep do the 2d convolution
    f = xarray_jax.vmap(lambda v: convolve2d(v.data, kernel, mode="same"), dim="time")
    return f(vis)


# Convolution with 1/n is just a smoother
kernel = jnp.ones((10, 10)) / 100
y_conv = conv2d(kernel, y)
plot(to_xr(y_conv, y), "smoothed")

# Back to autodiff, lets say we'd like to find "optimal kernel" for the
# smoothing. This could also be known as l2-regularized 2d-convolution
def conv2d_l2(kernel: jax.Array, y: xr.DataArray, lam: float = 1):
    pred = conv2d(kernel, y)
    # prediction error
    loss = jnp.square(y.data - pred).mean()
    # penalize complex kernels
    kernel_l2 = lam * jnp.square(kernel**2).mean()
    return loss + kernel_l2


# We can get the gradient and eventually do optimization
# like really it just works to take a gradient of scipy.convolve2d
d_conv = jax.grad(conv2d_l2)
grad = d_conv(kernel, y)
grad

# ..and it appears to work
loss_neg = float(conv2d_l2(kernel - grad * 1e-5, y)) # direction of the negative gradient
loss_cur = float(conv2d_l2(kernel, y))
loss_pos = float(conv2d_l2(kernel + grad * 1e-5, y)) # direction of the positive gradient

print(f"{loss_neg:.2f} < {loss_cur:.2f} < {loss_pos:.2f}")
