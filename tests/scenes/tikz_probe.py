"""TEST-ONLY probe: serialise a mosaickit Scene to TikZ.

mosaickit 0.5.1 ships no TikZ renderer (it was removed; domain packages own TikZ export),
so this is *not* a utility-viz feature. It exists to prove that the layers hold enough
geometry and resolved style to produce compilable TikZ, i.e. what a real exporter would
consume. Only the layer kinds used by the Cobb-Douglas slice are supported.
"""

from __future__ import annotations

from mosaickit import DEFAULT_PALETTE, Color, MarkerLayer, PathLayer, Scene, TextLayer, Theme

_DASH = {"solid": "", "dashed": "dashed", "dotted": "dotted", "dashdot": "dash dot"}


def _hex(color: Color | str, names: dict[str, str]) -> str:
    resolved = DEFAULT_PALETTE[color] if isinstance(color, str) and not color.startswith("#") else color
    value = resolved.to_hex(include_alpha=False) if isinstance(resolved, Color) else str(resolved)
    key = "c" + value.lstrip("#").upper()
    names[key] = value.lstrip("#").upper()
    return key


def scene_to_tikz(scene: Scene, theme: Theme) -> str:
    colors: dict[str, str] = {}
    body: list[str] = []
    for layer in scene.ordered_layers:
        if not layer.visible:
            continue
        bundle = theme.resolve(layer.role, fallback_category=layer.fallback_category)
        if isinstance(layer, PathLayer):
            stroke = layer.stroke.merged_over(bundle.stroke) if layer.stroke else bundle.stroke
            opts = [_hex(stroke.color, colors), f"line width={stroke.width}pt"]
            if _DASH[stroke.dash.value]:
                opts.append(_DASH[stroke.dash.value])
            if stroke.opacity is not None and stroke.opacity < 1:
                opts.append(f"opacity={stroke.opacity}")
            pts = " -- ".join(f"({x:.4f},{y:.4f})" for x, y in layer.path)
            body.append(f"\\draw[{','.join(opts)}] {pts};")
        elif isinstance(layer, MarkerLayer):
            marker = bundle.marker
            radius = (marker.size**0.5) / 2
            for x, y in layer.points:
                body.append(f"\\fill[{_hex(marker.color, colors)}] ({x:.4f},{y:.4f}) circle ({radius}pt);")
        elif isinstance(layer, TextLayer):
            style = layer.style.merged_over(bundle.text) if layer.style else bundle.text
            text = f"${layer.text}$" if layer.math else layer.text
            x, y = layer.position
            body.append(f"\\node[{_hex(style.color, colors)},anchor=south west] at ({x:.4f},{y:.4f}) {{{text}}};")
    defs = "\n".join(f"\\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in sorted(colors.items()))
    return (
        "\\documentclass[tikz,border=4pt]{standalone}\n\\usepackage{xcolor}\n"
        f"{defs}\n\\begin{{document}}\n\\begin{{tikzpicture}}\n"
        + "\n".join(body)
        + "\n\\end{tikzpicture}\n\\end{document}\n"
    )
