class ArgusError(Exception):
    """Base ARGUS exception."""


class ConfigurationError(ArgusError):
    """Configuration related exception."""


class BootError(ArgusError):
    """System boot exception."""


class ModelError(ArgusError):
    """Model related exception."""


class PipelineError(ArgusError):
    """Pipeline related exception."""