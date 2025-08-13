from importlib.metadata import PackageNotFoundError, version as _version
try:
    __version__ = _version("pyBRIXS")
except PackageNotFoundError:
    __version__ = "1.0.0"
