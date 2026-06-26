from typing import TYPE_CHECKING

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)


_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))
else:
    _import_structure["modular_blocks_gigaworld"] = ["GigaworldAutoBlocks"]
    _import_structure["modular_blocks_gigaworld_pyramid"] = ["GigaworldPyramidAutoBlocks"]
    _import_structure["modular_blocks_gigaworld_pyramid_distilled"] = ["GigaworldPyramidDistilledAutoBlocks"]
    _import_structure["modular_pipeline"] = [
        "GigaworldModularPipeline",
        "GigaworldPyramidDistilledModularPipeline",
        "GigaworldPyramidModularPipeline",
    ]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
    else:
        from .modular_blocks_gigaworld import GigaworldAutoBlocks
        from .modular_blocks_gigaworld_pyramid import GigaworldPyramidAutoBlocks
        from .modular_blocks_gigaworld_pyramid_distilled import GigaworldPyramidDistilledAutoBlocks
        from .modular_pipeline import (
            GigaworldModularPipeline,
            GigaworldPyramidDistilledModularPipeline,
            GigaworldPyramidModularPipeline,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
