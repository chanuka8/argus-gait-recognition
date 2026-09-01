class ArgusError(Exception):
    pass


class ConfigurationError(ArgusError):
    pass


class BootError(ArgusError):
    pass


class ModelError(ArgusError):
    pass


class PipelineError(ArgusError):
    pass
