"""Lazy matplotlib views of continuous geometry and archived GPU results."""

import numpy as np

from .constants import C0


def _axis(ax=None, polar=False):
    import matplotlib.pyplot as plt

    if ax is None:
        return plt.subplots(figsize=(7, 5), subplot_kw={"projection": "polar"} if polar else {})
    if polar != (getattr(ax, "name", "") == "polar"):
        raise ValueError("Axes projection does not match polar option")
    return ax.figure, ax


def _scale(config, units):
    if units == "m":
        return 1.0, "m"
    if units == "wavelength":
        return C0 / ((config["fmin"] + config["fmax"]) / 2), "λ₀"
    raise ValueError("units must be 'm' or 'wavelength'")


def geometry_plot(geometry, mesh, config, units, ax):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon, Rectangle

    fig, ax = _axis(ax)
    scale, label = _scale(config, units)
    lx, ly = np.array(geometry.size) / scale
    ax.set_facecolor("white")
    for shape in geometry.shapes:
        p = np.asarray(shape.parameters) / scale
        color = "#34495e" if shape.material == "PEC" else "white"
        if shape.kind == "circle":
            patch = Circle(p[:2], p[2])
        elif shape.kind == "rectangle":
            patch = Rectangle((p[0], p[2]), p[1] - p[0], p[3] - p[2])
        else:
            patch = Polygon(p, closed=True)
        patch.set(facecolor=color, edgecolor="none", zorder=2)
        ax.add_patch(patch)
    for key, color in (("tfsf_box", "#208b6d"), ("contour_box", "#b97518")):
        a, b, c, d = np.asarray(config["layout"][key]) / scale
        ax.add_patch(
            Rectangle(
                (a, c),
                b - a,
                d - c,
                fill=False,
                edgecolor=color,
                linestyle="--",
                label=key,
                zorder=4,
            )
        )
    px, py = (config["pml"][a]["thickness"] / scale for a in ("x", "y"))
    for x, y, w, h in (
        (0, 0, px, ly),
        (lx - px, 0, px, ly),
        (px, 0, lx - 2 * px, py),
        (px, ly - py, lx - 2 * px, py),
    ):
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#dce5eb", edgecolor="none", zorder=1))
    if mesh is not None:
        segments = [[(v / scale, 0), (v / scale, ly)] for v in mesh.x]
        segments += [[(0, v / scale), (lx, v / scale)] for v in mesh.y]
        ax.add_collection(
            LineCollection(segments, colors="#778899", linewidths=0.3, alpha=0.5, zorder=3)
        )
    ax.set(
        xlim=(0, lx),
        ylim=(0, ly),
        xlabel=f"x [{label}]",
        ylabel=f"y [{label}]",
        title="Exact geometry" + (" and Yee grid" if mesh is not None else ""),
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def mesh_plot(mesh, wavelength, units, ax):
    if mesh is None:
        raise RuntimeError("Call apply_mesh before plotting mesh")
    scale, label = _scale({"fmin": C0 / wavelength, "fmax": C0 / wavelength}, units)
    fig, ax = _axis(ax)
    for name, a in (("x", mesh.x), ("y", mesh.y)):
        ax.plot((a[:-1] + a[1:]) / (2 * scale), np.diff(a) / scale, label=name)
    ax.set(
        xlabel=f"Coordinate [{label}]",
        ylabel=f"Cell width [{label}]",
        title="Mesh spacing",
        ylim=(0, None),
    )
    ax.legend()
    return fig


def discretization_plot(geometry, mesh, coeff, config, units, ax):
    fig = geometry_plot(geometry, mesh, config, units, ax)
    ax = fig.axes[0] if ax is None else ax
    scale, _ = _scale(config, units)
    for op, xx, yy, full in (
        (coeff.hy, (mesh.x[:-1] + mesh.x[1:]) / 2, mesh.y, np.diff(mesh.x)[:, None]),
        (coeff.hx, mesh.x, (mesh.y[:-1] + mesh.y[1:]) / 2, np.diff(mesh.y)[None, :]),
    ):
        x, y = np.meshgrid(xx / scale, yy / scale, indexing="ij")
        cut = (op.length > 0) & (op.length < full * (1 - 1e-10))
        ax.scatter(x[cut], y[cut], s=9, color="#f29d38", zorder=5)
        for a, b, _ in op.pairs:
            ax.plot(x.ravel()[[a, b]], y.ravel()[[a, b]], color="#d33f49", lw=1.4, zorder=6)
    ax.set_title("Conformal cut faces (orange), enlarged pairs (red)")
    return fig


def source_plot(pulse, frequencies, ax):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    else:
        axes = np.asarray(ax).ravel()
        if len(axes) != 2:
            raise ValueError("Source plot requires two axes")
        fig = axes[0].figure
    t = np.linspace(0, pulse.duration, 4000)
    axes[0].plot(t * 1e9, pulse(t))
    axes[0].set(xlabel="Time [ns]", ylabel="Source amplitude", title="Gaussian modulated sinusoid")
    f = np.linspace(0, 1.1 * pulse.significant_frequency, 1000)
    axes[1].plot(f / 1e9, 20 * np.log10(np.maximum(pulse.spectrum(f), 1e-12)))
    axes[1].scatter(
        frequencies / 1e9, 20 * np.log10(pulse.spectrum(frequencies)), s=15, label="DFT bins"
    )
    axes[1].set(
        xlabel="Frequency [GHz]",
        ylabel="Designed amplitude [dB]",
        ylim=(-60, 2),
        title="Continuous pulse spectrum",
    )
    axes[1].legend()
    fig.tight_layout()
    return fig


def convergence_plot(result, per_bin, ax):
    fig, ax = _axis(ax)
    h = result.convergence
    if per_bin:
        import matplotlib.pyplot as plt

        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(result.frequencies)))
        values = np.maximum(h.current_error_ratio, h.incident_error_ratio)
        for i, (f, c) in enumerate(zip(result.frequencies, colors)):
            ax.semilogy(
                h.steps,
                np.maximum(values[:, i], 1e-16),
                color=c,
                alpha=0.8,
                label=f"{f / 1e9:g} GHz" if len(colors) <= 7 else None,
            )
        if len(colors) > 7:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            fig.colorbar(
                ScalarMappable(
                    norm=Normalize(result.frequencies[0] / 1e9, result.frequencies[-1] / 1e9),
                    cmap="viridis",
                ),
                ax=ax,
                label="Frequency [GHz]",
            )
    else:
        ax.semilogy(h.steps, np.maximum(result.history[:, 2], 1e-16), label="Worst DFT bin")
    tolerance = result.configuration["settings"]["stop"]["field_tol"]
    ax.semilogy(
        h.steps,
        np.maximum(h.residual / tolerance, 1e-16),
        "--",
        color="#d05b37",
        label="Field residual / tolerance",
    )
    ax.axhline(1, color="black", lw=0.8, label="Acceptance threshold")
    ax.set(
        xlabel="FDTD step",
        ylabel="Error / tolerance",
        title=f"GPU convergence: {result.diagnostics['status']}",
    )
    ax.legend(fontsize=8)
    return fig


def _frequency(result, frequency):
    if frequency is None:
        return len(result.frequencies) // 2
    indices = np.flatnonzero(np.isclose(result.frequencies, frequency, rtol=1e-10, atol=0))
    if len(indices) != 1:
        raise ValueError("frequency must match one stored DFT bin in Hz")
    return int(indices[0])


def scattering_plot(result, frequency, scale, normalize, polar, ax):
    i = _frequency(result, frequency)
    width = result.scattering_width[i]
    if normalize == "wavelength":
        width, label = width / (C0 / result.frequencies[i]), "σ₂D / λ"
    elif normalize == "metres":
        label = "σ₂D [m]"
    else:
        raise ValueError("normalize must be 'wavelength' or 'metres'")
    if scale == "db":
        width = 10 * np.log10(np.maximum(width, 1e-30))
        label = (
            "10 log₁₀(σ₂D / λ) [dB]" if normalize == "wavelength" else "10 log₁₀(σ₂D / 1 m) [dB]"
        )
    elif scale != "linear":
        raise ValueError("scale must be 'linear' or 'db'")
    fig, ax = _axis(ax, polar)
    ax.plot(result.angles if polar else np.rad2deg(result.angles), width)
    ax.set(title=f"2D scattering width · {result.frequencies[i] / 1e9:g} GHz", ylabel=label)
    if not polar:
        ax.set_xlabel("Scattering angle [deg]")
    return fig


def far_field_plot(result, frequency, component, ax):
    i = _frequency(result, frequency)
    s = result.far_field[i]
    fig, ax = _axis(ax)
    angle = np.rad2deg(result.angles)
    if component == "complex":
        ax.plot(s.real, s.imag)
        ax.set(xlabel="Re S", ylabel="Im S")
        ax.set_aspect("equal", adjustable="datalim")
    else:
        if component == "magnitude":
            y, label = abs(s), "|S|"
        elif component == "real":
            y, label = s.real, "Re S"
        elif component == "imag":
            y, label = s.imag, "Im S"
        elif component == "phase":
            y = np.where(abs(s) > max(abs(s).max() * 1e-8, 1e-30), np.angle(s), np.nan)
            label = "arg S [rad] (nulls masked)"
        else:
            raise ValueError("component must be magnitude, real, imag, phase or complex")
        ax.plot(angle, y)
        ax.set(xlabel="Scattering angle [deg]", ylabel=label)
    ax.set_title(f"Complex far field · {result.frequencies[i] / 1e9:g} GHz")
    return fig
