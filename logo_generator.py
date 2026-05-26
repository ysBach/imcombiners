from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

BG_COLOR = "#0D1117"
PANEL_COLOR = "#101820"
PYTHON_BLUE = "#4B8BBE"
CYAN_GREEN = "#24C8A6"
RUST_ORANGE = "#CE412B"
STAR_YELLOW = "#FFE873"
STAR_CORE = "#FFF8C5"


def _add_rounded_rect(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    edgecolor: str,
    facecolor: str,
    linewidth: float = 2.4,
    alpha: float = 1.0,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _add_tile_grid(
    ax,
    patch: FancyBboxPatch,
    xy: tuple[float, float],
    size: float,
    *,
    color: str = "white",
    alpha: float = 0.16,
    zorder: int = 2,
) -> None:
    x0, y0 = xy
    step = size / 10
    for idx in range(1, 15):
        x = x0 + idx * step
        y = y0 + idx * step
        vline = Line2D(
            [x, x],
            [y0, y0 + size],
            color=color,
            linewidth=0.45,
            alpha=alpha,
            zorder=zorder,
        )
        hline = Line2D(
            [x0, x0 + size],
            [y, y],
            color=color,
            linewidth=0.45,
            alpha=alpha,
            zorder=zorder,
        )
        vline.set_clip_path(patch)
        hline.set_clip_path(patch)
        ax.add_line(vline)
        ax.add_line(hline)


def _add_star(
    ax,
    tile_xy: tuple[float, float],
    tile_size: float,
    pos: tuple[float, float],
    radius: float,
    *,
    output: bool = False,
    zorder: int = 5,
) -> None:
    x = tile_xy[0] + pos[0] * tile_size
    y = tile_xy[1] + pos[1] * tile_size
    glow_color = STAR_YELLOW if output else "#FFFFFF"
    core_color = STAR_CORE if output else "#FFFFFF"
    for scale, alpha in ((1.0, 0.16), (0.9, 0.24)):
        ax.add_patch(
            Circle(
                (x, y),
                radius * scale,
                facecolor=glow_color,
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
            )
        )
    ax.add_patch(
        Circle(
            (x, y),
            radius * 0.75,
            facecolor=core_color,
            edgecolor="none",
            alpha=0.96,
            zorder=zorder + 1,
        )
    )


def _add_artifact_dot(
    ax,
    tile_xy: tuple[float, float],
    tile_size: float,
    pos: tuple[float, float],
    radius: float,
) -> None:
    x = tile_xy[0] + pos[0] * tile_size
    y = tile_xy[1] + pos[1] * tile_size
    ax.add_patch(
        Circle(
            (x, y),
            radius,
            facecolor=RUST_ORANGE,
            edgecolor="none",
            alpha=0.98,
            zorder=8,
        )
    )


def _add_artifact_streak(
    ax,
    tile_xy: tuple[float, float],
    tile_size: float,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linewidth: float,
) -> None:
    x0 = tile_xy[0] + start[0] * tile_size
    y0 = tile_xy[1] + start[1] * tile_size
    x1 = tile_xy[0] + end[0] * tile_size
    y1 = tile_xy[1] + end[1] * tile_size
    ax.plot(
        [x0, x1],
        [y0, y1],
        color=RUST_ORANGE,
        linewidth=linewidth,
        alpha=0.98,
        solid_capstyle="round",
        zorder=8,
    )


def _add_tile(
    ax,
    xy: tuple[float, float],
    size: float,
    *,
    edgecolor: str,
    fillcolor: str,
    output: bool = False,
    artifact: str | None = None,
    zorder: int = 2,
) -> None:
    patch = _add_rounded_rect(
        ax,
        xy,
        size,
        size,
        edgecolor=edgecolor,
        facecolor=fillcolor,
        linewidth=2.8 if output else 2.2,
        zorder=zorder,
    )
    _add_tile_grid(
        ax,
        patch,
        xy,
        size,
        alpha=0.18 if output else 0.13,
        zorder=zorder + 1,
    )

    stars = [
        ((0.34, 0.72), 0.08),
        ((0.66, 0.60), 0.1),
        ((0.48, 0.30), 0.13),
        ((0.76, 0.27), 0.06),
    ]
    for pos, radius in stars:
        _add_star(ax, xy, size, pos, radius * size, output=output, zorder=zorder + 3)

    if artifact == "dot-upper":
        _add_artifact_dot(ax, xy, size, (0.70, 0.84), 0.050 * size)
    elif artifact == "dot-left":
        _add_artifact_dot(ax, xy, size, (0.16, 0.51), 0.050 * size)
    elif artifact == "streak-lower":
        _add_artifact_streak(
            ax,
            xy,
            size,
            (0.16, 0.28),
            (0.53, 0.20),
            linewidth=0.25 * size * 72,
        )
    elif artifact == "streak-upper":
        _add_artifact_streak(
            ax,
            xy,
            size,
            (0.51, 0.86),
            (0.89, 0.77),
            linewidth=0.25 * size * 72,
        )


def generate_imcombiners_logo(
    output_filename: str | Path = "logo.png",
    *,
    dpi: int = 300,
) -> Path:
    """Generate the imcombiners logo as a PNG image.

    The mark shows four same-footprint bluish input arrays containing the same
    stationary stars plus transient red artifacts. The central yellow output
    array keeps the stationary stars and omits the artifacts, matching ordinary
    imcombine-style stack reduction without implying affine registration.
    """
    output_path = Path(output_filename)

    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=dpi)
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    _add_rounded_rect(
        ax,
        (0.02, 0.02),
        0.96,
        0.96,
        edgecolor="none",
        facecolor=BG_COLOR,
        linewidth=0,
        zorder=0,
    )

    input_size = 0.28
    output_size = 0.40
    input_tiles = [
        ((0.07, 0.65), "#4B8BBE", "#183149", "dot-upper"),
        ((0.65, 0.65), "#5294BE", "#183447", "streak-lower"),
        ((0.07, 0.07), "#4484B8", "#172F45", "dot-left"),
        ((0.65, 0.07), "#589EBE", "#18374A", "streak-upper"),
    ]
    for xy, edgecolor, fillcolor, artifact in input_tiles:
        _add_tile(
            ax,
            xy,
            input_size,
            edgecolor=edgecolor,
            fillcolor=fillcolor,
            artifact=artifact,
            zorder=2,
        )

    _add_tile(
        ax,
        (0.30, 0.30),
        output_size,
        edgecolor=STAR_YELLOW,
        fillcolor="#3A3C2A",
        output=True,
        zorder=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="png",
        dpi=dpi,
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_imcombiners_logo()
