import jax
import jax.numpy as jnp
import xarray as xr
from flax import nnx
from figs import plot, to_xr, fill_na, fill_na_all
from tqdm import tqdm
import optax
from itertools import batched


# (you need to adjust the dates)
# https://opendata.fmi.fi/download?producer=harmonie_scandinavia_surface&param=Pressure,GeopHeight,Temperature,DewPoint,Humidity,WindDirection,WindSpeedMS,WindUMS,WindVMS,PrecipitationAmount,CAPE,TotalCloudCover,LowCloudCover,MediumCloudCover,HighCloudCover,RadiationGlobal,Visibility,WindGust,RadiationGlobalAccumulation,RadiationNetSurfaceLWAccumulation,RadiationNetSurfaceSWAccumulation,RadiationSWAccumulation&bbox=18,55,35,74&origintime=2026-06-08T06:00:00Z&starttime=2026-06-08T06:00:00Z&endtime=2026-06-11T00:00:00Z&format=netcdf&projection=EPSG:4326&levels=0&timestep=60
ds = xr.open_dataset("data/harmonie.nc").drop_dims("time_h")

# Visibility in km, 349 null values, fix somehow
y = fill_na(ds["visibility_in_air_407"] / 1000)
x = ds.drop_vars(("visibility_in_air_407", "crs")).to_array()
x = fill_na_all(x.transpose("time", "lat", "lon", "variable"))
n, batch_size = 20, 4

rngs = nnx.Rngs(0)


# Note that while this convolutional neural network in itself might make
# sense, the data certainly does not! We are training the model against
# forecasts of different forecast window and then verifying the perdiction
# against the analysis...
class CNet(nnx.Module):
    def __init__(self, d_in: int, d_mid: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=d_in, out_features=d_mid, kernel_size=(5, 5), padding="SAME", rngs=rngs)
        self.norm1 = nnx.LayerNorm(num_features=d_mid, rngs=rngs)
        self.out = nnx.Conv(in_features=d_mid, out_features=1, kernel_size=(1, 1), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        return jax.nn.relu(self.out(x)[..., 0])


# Define the model and the optimizer
model = CNet(d_in=x.shape[3], d_mid=32, rngs=rngs)

opt = nnx.Optimizer(model, optax.adamw(learning_rate=1e-1), wrt=nnx.Param)


# mse loss, we also collect the score, which in this case is rmse scaled
# by the mean of the target
def rmse_loss(model, x, y):
    loss = optax.squared_error(model(x), y).mean()
    score = jnp.sqrt(loss) / y.mean()
    return loss, score


# execute one training step, jax.jit compiles this ahead of time so its faster
# during execution
@nnx.jit
def train_step(model, optim, x, y):
    grad_fn = nnx.value_and_grad(rmse_loss, has_aux=True)
    loss_scores, grads = grad_fn(model, x, y)
    optim.update(model, grads)
    return loss_scores

# do one epoch of training and report losses - of course here we have very
# litte data so one epoch is quite fast
def train_epoch(model, train_step, opt, epoch, x, y) -> tuple[float, float]:
    train_loss, train_score = 0.0, 0.0
    # take batches of 4, a better (production) way would be to use a real dataloader
    for inds in tqdm(batched(range(n), 4), desc=f"Epoch {epoch}", leave=False):
        batch = list(inds)
        loss, score = train_step(model, opt, x[batch, :, :, :], y[batch, :, :])
        train_loss += loss
        train_score += score
    return train_loss * batch_size / n, train_score * batch_size / n


# do 20 epochs just to see we really are training
for epoch in range(20):
    train_loss, train_score = train_epoch(model, train_step, opt, epoch, x.data, y.data)
    print(f"[Epoch {epoch:2d}], Training score: {train_score:05.2f},")


y_pred = model(x.data[[0]])
plot(to_xr(y_pred, y[[0]]), "predicted")
