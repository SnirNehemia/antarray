import typing as tp
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple, get_args
import time
import jax
import jax.numpy as jnp
import numpy as np
from joblib import Memory
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.projections import PolarAxes

import physics
from tapering import hamming_taper, ideal_steering, uniform_taper

root_dir = Path(__file__).parent.parent
memory_dir = root_dir / ".joblib_cache"
memory = Memory(memory_dir, mmap_mode="r", verbose=0)  # Disk cache for CST files


@memory.cache
def load_cst_file(cst_path: Path) -> np.ndarray:
    """Load CST antenna pattern data from a file."""
    names = (
        "elev_deg",
        "azim_deg",
        "abs_grlz",
        "abs_cross",
        "phase_cross_deg",
        "abs_copol",
        "phase_copol_deg",
        "ax_ratio",
    )
    data = np.genfromtxt(cst_path, skip_header=2, dtype=np.float32, names=names)
    data = data[np.argsort(data, order=["elev_deg", "azim_deg"])]

    phase_cross = np.radians(data["phase_cross_deg"])
    phase_copol = np.radians(data["phase_copol_deg"])
    E_cross = np.sqrt(data["abs_cross"]) * np.exp(1j * phase_cross)
    E_copol = np.sqrt(data["abs_copol"]) * np.exp(1j * phase_copol)
    field = np.stack([E_cross, E_copol], axis=-1)

    field = field.reshape(181, 360, 2)  # (elev, azim, n_pol)
    field = field[:-1]  #  Remove last elev value
    return field


@memory.cache
def load_cst(cst_path: Path) -> np.ndarray:
    print(f"Loading antenna pattern from {cst_path}")
    data = {}
    for path in cst_path.iterdir():
        if "_RG.txt" not in path.name:
            continue  # Skip non-RG files

        i = int(path.stem.split("[")[1].split("]")[0]) - 1
        data[i] = load_cst_file(path)

    data = [v for _, v in sorted(data.items())]
    fields = np.stack(data, axis=0)  # (n_elements, elev, azim, n_pol)
    fields = fields.reshape(4, 4, *fields.shape[1:])  # (n_y, n_x, elev, azim, n_pol)
    fields = fields.transpose(1, 0, 2, 3, 4)  # (n_x, n_y, elev, azim, n_pol)
    return fields


def argmax_nd(x: jax.Array) -> tuple[int, ...]:
    indices = np.unravel_index(np.argmax(x), x.shape)
    indices = tuple(i.item() for i in indices)
    return indices


def to_power(x: jax.Array) -> jax.Array:
    x = jnp.abs(x) ** 2  # Convert to power (magnitude squared)
    x = jnp.sum(x, axis=-1)  # Sum over polarization
    return x


def to_db(x: jax.Array) -> jax.Array:
    return 10 * jnp.log10(x)  # Convert to dB


def to_power_db(x: jax.Array) -> jax.Array:
    return to_db(to_power(x))


def plot_power_db(power_db: jax.Array, title: str | None = None, fig=None) -> None:
    indices = argmax_nd(power_db)
    fig = physics.plot_pattern(power_db, clip_min_db=-40.0, fig=fig)
    if title:
        title = f"{title} | Peak {power_db[indices]:.3f} at {indices=}"
        fig.suptitle(title)


def mse_loss(pred_power: jax.Array, target_power: jax.Array) -> jax.Array:
    return jnp.mean((pred_power - target_power) ** 2)


LossFn = tp.Callable[[jax.Array, jax.Array], jax.Array]
LossScale = tp.Literal["linear", "db"]


@partial(jax.jit, static_argnames=("loss_fn", "loss_scale"))
def train_step(
    w: jnp.ndarray,
    aeps: jax.Array,
    target_power: jax.Array,
    lr: float,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
) -> tuple[jnp.ndarray, float]:
    if loss_scale == "db":
        target_power = to_db(target_power)

    def loss_wrapper(w: jax.Array) -> jax.Array:
        pred_pattern = jnp.einsum("xy,xytpz->tpz", w, aeps)
        pred_power = to_power(pred_pattern)
        if loss_scale == "db":
            pred_power = to_db(pred_power)
        return loss_fn(pred_power, target_power)

    loss, grads = jax.value_and_grad(loss_wrapper)(w)
    w = w - lr * grads
    # w = w / jnp.linalg.norm(w) * amplitude  # Re-normalize
    # amplitude = jnp.abs(w)
    # norm_factor = np.sqrt(np.sum(amplitude**2))
    # w = w / norm_factor
    # w = my_norm(w)
    return w, loss


def optimize(
    w_init: jax.Array,
    aeps: jax.Array,
    target_power: jax.Array,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
    lr: float = 1e-5,
    iter_num = 200,
    global_min = True
) -> jax.Array:
    w = w_init
    min_loss = 1e3
    for step in range(iter_num):
        w, loss = train_step(w, aeps, target_power, lr, loss_fn, loss_scale)
        w = my_norm(w)
        if global_min and loss < min_loss:
            min_loss = loss
            min_w = w
            pattern_opt = jnp.einsum("xy,xytpz->tpz", w, aeps)
            min_pattern = to_power_db(pattern_opt)
        if step % 10 == 0:
            amplitude = np.abs(w)
            norm_factor = np.sqrt(np.sum(amplitude**2))
            print(f"step {step}, loss: {loss:.3f} | w sum: {norm_factor:.2f}")

    print(f"Weight norm: {jnp.linalg.norm(w):.3f}")

    
    if global_min:
        return min_pattern, min_w
    else:
        pattern_opt = jnp.einsum("xy,xytpz->tpz", w, aeps)
        power_db_opt = to_power_db(pattern_opt)
        return power_db_opt, w


Taper = Literal["hamming", "uniform"]


class OptParams(NamedTuple):
    taper: Taper
    elev_deg: float
    azim_deg: float
    w: jax.Array
    aeps_env0: jax.Array
    aeps_env1: jax.Array
    power_env0: jax.Array
    power_env1: jax.Array


def init_params(
    env0_name: str = "no_env_rotated",
    env1_name: str = "Env1_rotated",
    taper: Taper = "uniform",
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
) -> OptParams:
    cst_dir = root_dir / "cst"
    aeps_env0 = load_cst(cst_dir / env0_name)

    # Compute weights
    nx, ny = aeps_env0.shape[:2]
    if taper == "hamming":
        amplitude = hamming_taper(nx=nx, ny=ny)
    else:
        amplitude = uniform_taper(nx=nx, ny=ny)
    amplitude = amplitude / np.sqrt(np.sum(amplitude**2))  # Normalize power to 1

    phase = ideal_steering(nx=nx, ny=ny, elev_deg=elev_deg, azim_deg=azim_deg)
    w = amplitude * np.exp(1j * phase)

    power_env0 = to_power(np.einsum("xy,xytpz->tpz", w, aeps_env0))

    aeps_env1 = load_cst(cst_dir / env1_name)
    power_env1 = to_power(np.einsum("xy,xytpz->tpz", w, aeps_env1))

    return OptParams(
        taper=taper,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        w=jnp.asarray(w),
        aeps_env0=jnp.asarray(aeps_env0),
        aeps_env1=jnp.asarray(aeps_env1),
        power_env0=jnp.asarray(power_env0),
        power_env1=jnp.asarray(power_env1),
    )


class OptResults(NamedTuple):
    power_db_opt: jax.Array
    nmse_env1: float
    nmse_opt: float

def my_norm(w):
    amplitude = np.abs(w)
    norm_factor = np.sqrt(np.sum(amplitude**2))
    w = w / norm_factor
    return w

# @memory.cache
def run_optimization(
    env0_name: str = "no_env_rotated",
    env1_name: str = "Env1_rotated",
    taper: Taper = "uniform",
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
    lr: float = 1e-5,
    plot: bool = True,
    init_w = 'taper',
    iter_num = 200,
    global_min = True
):
    print(f"Running optimization for elev {elev_deg}°, azim {azim_deg}°, {taper} taper")
    params = init_params(
        taper=taper,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        env0_name=env0_name,
        env1_name=env1_name,
    )
    w, aeps_env1, power_env0 = params.w, params.aeps_env1, params.power_env0
    if init_w != 'taper':
        cst_dir = root_dir / "cst"
        aeps_env0 = load_cst(cst_dir / env0_name)
        nx, ny = aeps_env0.shape[:2]
        if init_w == "hamming":
            print(f' -- using {init_w:s} taper as init weight -- ')
            amplitude = hamming_taper(nx=nx, ny=ny)
        elif init_w == 'uniform':
            print(f' -- using {init_w:s} taper as init weight -- ')
            # amplitude = uniform_taper(nx=nx, ny=ny)
            amplitude = np.ones([nx, ny])
            # print(f'normalized amplitude sum is: {np.sum(amplitude**2):.2f}')
            phase = np.zeros([nx, ny])
            w = my_norm(amplitude * np.exp(1j * phase))
        elif init_w == 'random':
            print(f' -- using {init_w:s} taper as init weight -- ')
            amplitude = np.random.rand(nx, ny)
            # print(f'normalized amplitude sum is: {np.sum(amplitude**2):.2f}')
            phase = np.random.rand(nx, ny) * 2 * np.pi - np.pi
            w = my_norm(amplitude * np.exp(1j * phase))
        else:
            print(f'did not find matching to {init_w:s}')
    power_db_opt, w = optimize(w, aeps_env1, power_env0, loss_fn, loss_scale, lr, iter_num, global_min=global_min)

    power_db_env0, power_db_env1 = to_db(params.power_env0), to_db(params.power_env1)

    # Normalize by the mean square of the target pattern in dB
    norm_factor = np.mean(power_db_env0**2)
    mse_env1_raw = np.mean((power_db_env1 - power_db_env0) ** 2)
    mse_opt_raw = np.mean((power_db_opt - power_db_env0) ** 2)

    # Calculate NMSE in dB
    nmse_env1 = 10 * np.log10(mse_env1_raw / norm_factor).item()
    nmse_opt = 10 * np.log10(mse_opt_raw / norm_factor).item()

    if plot:
        fig = plt.figure(figsize=(15, 12), layout="compressed")
        subfigs = fig.subfigures(3, 1)

        plot_power_db(power_db_env0, fig=subfigs[0], title="No Env")

        title = f"Env 1 | NMSE {nmse_env1:.3f}dB"
        plot_power_db(power_db_env1, fig=subfigs[1], title=title)

        title = f"Optimized | NMSE {nmse_opt:.3f}dB"
        plot_power_db(power_db_opt, fig=subfigs[2], title=title)

        steer_title = f"Steering Elev {int(elev_deg)}°, Azim {int(azim_deg)}°"
        title = f"{taper} taper | {steer_title}"
        fig.suptitle(title, fontsize=16)

        name = f"patterns_{taper}_elev_{int(elev_deg)}_azim_{int(azim_deg)}.png"
        fig.savefig(name, dpi=200)
        print(f"Saved figure to {name}")

        # Additional weights visualization (magnitude and phase)
        try:
            w_np = np.asarray(w)
            w_np_norm = w_np * np.exp(-1j*np.angle(w_np[1][1]))
            figW, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(9, 4),
                layout="compressed",
            )

            im1 = ax1.imshow(np.abs(w_np_norm), origin="upper")
            ax1.set_aspect("equal")
            figW.colorbar(im1, ax=ax1)
            ax1.set_title("w magnitude")
            ax1.set_xlabel("x index")
            ax1.set_ylabel("y index")

            phase_deg = np.degrees(np.angle(w_np_norm))
            im2 = ax2.imshow(
                phase_deg,
                origin="upper",
                cmap="hsv",
                vmin=-180,
                vmax=180,
            )
            ax2.set_aspect("equal")
            figW.colorbar(im2, ax=ax2)
            ax2.set_title("w phase (deg)")
            ax2.set_xlabel("x index")
            ax2.set_ylabel("y index")

            nameW = f"weights_{taper}_elev_{int(elev_deg)}_azim_{int(azim_deg)}.png"
            figW.savefig(nameW, dpi=200)
            print(f"Saved weights figure to {nameW}")
        except Exception as e:
            print(f"Failed to plot weights: {e}")

    return [OptResults(
        power_db_opt=power_db_opt,
        nmse_env1=nmse_env1,
        nmse_opt=nmse_opt,
    ), w]


# run_optimization(
#     taper="hamming",
#     elev_deg=20,
#     azim_deg=-45,
#     loss_scale="db",
#     lr=5e-6,
# )


def evaluate_grid(
    env0_name: str = "no_env_rotated",
    env1_name: str = "Env1_rotated",
    taper: Taper = "hamming",
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
    lr: float = 5e-6,
) -> None:
    elevs = np.arange(0, 30, 10)
    azims = np.arange(0, 360, 30)
    results = {}

    opt = partial(
        run_optimization,
        env0_name=env0_name,
        env1_name=env1_name,
        taper=taper,
        loss_fn=loss_fn,
        loss_scale=loss_scale,
        lr=lr,
        plot=False,
    )

    for elev in elevs:
        for azim in azims:
            elev_deg, azim_deg = elev.item(), azim.item()
            res, _ = opt(elev_deg=elev_deg, azim_deg=azim_deg)
            results[(elev, azim)] = res

    nmses_env1 = np.array([res.nmse_env1 for res in results.values()]).reshape(
        len(elevs), len(azims)
    )
    nmses_opt = np.array([res.nmse_opt for res in results.values()]).reshape(
        len(elevs), len(azims)
    )

    nmse_diff = nmses_opt - nmses_env1
    print(f"Mean NMSE Improvement: {nmse_diff.mean():.3f} dB")
    print(f"Std NMSE Improvement: {nmse_diff.std():.3f} dB")
    print(f"Max NMSE Improvement: {nmse_diff.max():.3f} dB")
    print(f"Min NMSE Improvement: {nmse_diff.min():.3f} dB")

    basic_plot = False
    if basic_plot:
        fig, ax = plt.subplots(figsize=(15, 5), layout="compressed")
        im2 = ax.imshow(nmse_diff, cmap="viridis", origin="lower")
        ax.set_xticks(np.arange(len(azims)), labels=azims)
        ax.set_yticks(np.arange(len(elevs)), labels=elevs)
        ax.set(title="NMSE Improvement", xlabel="Azim (deg)", ylabel="Elev (deg)")
        fig.colorbar(im2, ax=ax, label="NMSE (dB)")
        name = f"patterns_{taper}_nmse_grid.png"
        fig.savefig(name, dpi=200)

    polar_plot = True
    if polar_plot:
        kw = dict(layout="compressed", subplot_kw={"projection": "polar"})
        fig, ax = plt.subplots(figsize=(8, 8), **kw)
        axp = tp.cast(PolarAxes, ax)  # For type checker
        azim_grid, elev_grid = np.meshgrid(np.radians(azims), elevs)
        kw = dict(cmap="RdBu", norm=colors.CenteredNorm())  # Center colormap at 0
        c = axp.pcolormesh(azim_grid, elev_grid, nmse_diff, **kw)
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        fig.colorbar(c, ax=axp, label="NMSE (dB)")

        envs = f"{env0_name} vs {env1_name}"
        loss_fn_name = loss_fn.__name__.replace("_", " ")

        title = f"ΔNMSE | {taper} taper | {envs} | {loss_fn_name} ({loss_scale}) | {lr=:.1e}"
        fig.suptitle(title)

        envs = envs.replace(" ", "_")
        loss_fn_name = loss_fn.__name__
        name = (
            f"patterns_nmse_{taper}_{envs}_{loss_fn_name}_{loss_scale}_lr_{lr:.1e}.png"
        )
        fig.savefig(name, dpi=200)

tic = time.time()

# #  ----------------- TEST SINGLE OPTIMIZATION -----------------
print('run single optimization')
# env0_name = "no_env_rotated"
# env1_name = "Env1_rotated"
# taper = "hamming"
# elev_deg = 40.0
# azim_deg = 40.0
# loss_fn = mse_loss
# loss_scale = "linear"
# lr = 1e-5
# plot = True
azim_degs = [0] # [0,90,180,270]
for azim_deg in azim_degs:
    [temp, w] = run_optimization(
        env0_name = "no_env_rotated",
        env1_name = "Env3_1_rotated",
        taper = "uniform",  # 'uniform' | 'hamming'
        elev_deg = 0.0,
        azim_deg = azim_deg,
        loss_fn = mse_loss,
        loss_scale = "linear",
        lr = 1e-5,
        plot = True,
        init_w = 'uniform', # 'taper' , 'uniform', 'random', 'hamming'
        iter_num = 5000,
        global_min = True
        )



# ----------------- TEST FULL OPTIMIZATION -----------------

# env = "Env1_rotated"
# taper = "hamming"
# loss_fn = mse_loss
# loss_scale = 'linear'
# lr = 1e-6
# evaluate_grid(
#     env1_name=env,
#     taper=taper,
#     loss_fn=loss_fn,
#     loss_scale=loss_scale,
#     lr=lr,
# )

# -------------------------------------------------------------------------------------

print(f'total run time: {time.time()-tic:.0f} sec = {(time.time()-tic)/60:.0f} min')
print('done')

# for env in [
#     "Env1_rotated",
#     "Env2_rotated",
#     "Env1_1_rotated",
#     "Env1_2_rotated",
#     "Env2_1_rotated",
#     "Env2_2_rotated",
# ]:
#     for taper in get_args(Taper):
#         for loss_fn in [mse_loss]:
#             for loss_scale in tp.get_args(LossScale):
#                 for lr in [1e-5, 5e-6]:
#                     evaluate_grid(
#                         env1_name=env,
#                         taper=taper,
#                         loss_fn=loss_fn,
#                         loss_scale=loss_scale,
#                         lr=lr,
#                     )
