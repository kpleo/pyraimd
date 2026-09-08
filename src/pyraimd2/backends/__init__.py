"""Backend factories: builtin names, entry-point plugins, capability checks."""

from pyraimd2.backends.registry import (
    ENTRY_POINT_GROUP,
    BackendRegistration,
    BackendRegistryError,
    assert_capabilities_satisfy,
    available_backends,
    backend_capabilities,
    backend_factory,
    create_backend,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "BackendRegistration",
    "BackendRegistryError",
    "assert_capabilities_satisfy",
    "available_backends",
    "backend_capabilities",
    "backend_factory",
    "create_backend",
]
