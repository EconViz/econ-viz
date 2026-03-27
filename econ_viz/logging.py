"""
Centralised logging configuration for econ-viz.

Every module should obtain its logger via::

    from econ_viz.logging import get_logger
    logger = get_logger(__name__)

By default the logger is set to ``WARNING`` so end-users see nothing.
Developers can lower the threshold to ``DEBUG`` for full numerical traces::

    import logging
    logging.getLogger("econ_viz").setLevel(logging.DEBUG)
"""

import logging

_LIBRARY_ROOT = "econ_viz"

# Attach a NullHandler so that library consumers who have not configured
# logging do not see "No handler found" warnings.
logging.getLogger(_LIBRARY_ROOT).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``econ_viz`` namespace.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.  If it already starts
        with ``econ_viz.`` it is used as-is; otherwise it is prefixed.

    Returns
    -------
    logging.Logger
    """
    if not name.startswith(_LIBRARY_ROOT):
        name = f"{_LIBRARY_ROOT}.{name}"
    return logging.getLogger(name)
