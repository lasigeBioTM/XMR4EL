import logging

from typing import Optional

_pkg_name = "xmr4el"
_pkg_logger = logging.getLogger(_pkg_name)

def _ensure_default_handler():
    """
    Add a sensible StreamHandler only if the package logger has no handlers
    and no ancestor configured handlers (prevents noisy double-handling).
    """
    # If any handler exists on logger or its parents, don't add one
    if _pkg_logger.handlers:
        return

    # If any root-level handlers configured (basicConfig or app), do nothing
    if logging.getLogger().handlers:
        return

    handler = logging.StreamHandler()
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    _pkg_logger.addHandler(handler)
    _pkg_logger.setLevel(logging.INFO)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    _ensure_default_handler()
    return _pkg_logger.getChild(name) if name else _pkg_logger

def set_logger(logger: logging.Logger):
    """Replace package logger (package-level override)."""
    global _pkg_logger
    _pkg_logger = logger

def set_verbosity(level: int):
    """0 = WARNING, 1 = INFO, 2+ = DEBUG"""
    mapping = [logging.WARNING, logging.INFO, logging.DEBUG]
    lvl = mapping[min(max(level, 0), len(mapping) - 1)]
    logging.getLogger(_pkg_name).setLevel(lvl)
    get_logger().info("xmr4el verbosity set to %s", logging.getLevelName(lvl))