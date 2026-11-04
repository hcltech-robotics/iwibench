"""IWIBench and its optional Verifiers v1 taskset."""

__version__ = "0.2.5"
__all__ = ["IWIBenchTaskset", "IWIBenchEnv"]


def __getattr__(name: str):
    if name in __all__:
        from iwibench import taskset

        value = getattr(taskset, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
