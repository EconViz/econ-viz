# econ-viz

`econ-viz` was renamed to **[utility-viz](https://pypi.org/project/utility-viz/)** in 2.0.

This package depends on `utility-viz` (the same version) and adds the compatibility layer for
existing 1.x code, throughout 2.x: the `econ_viz` import package and the `econ-viz` command. Using
either emits a deprecation warning. Both are removed in 3.0.

```bash
pip install utility-viz          # use this name from now on (no econ_viz, no econ-viz command)
```

See the migration guide: https://github.com/EconViz/econ-viz#migrating-from-econ-viz
