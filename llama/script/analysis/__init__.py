__all__ = ["main"]


def __getattr__(name):
    if name == "main":
        from .runner import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
