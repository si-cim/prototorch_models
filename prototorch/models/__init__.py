from importlib.metadata import PackageNotFoundError, version

VERSION_FALLBACK = "uninstalled_version"
try:
    __version__ = version(__name__.replace(".", "-"))
except PackageNotFoundError:
    __version__ = VERSION_FALLBACK
    pass
