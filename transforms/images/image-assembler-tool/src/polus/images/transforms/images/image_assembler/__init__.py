"""Image Assembler Plugin."""

__version__ = "1.4.1-dev0"

from polus.plugins.transforms.images.image_assembler.image_assembler import (  # noqa
    assemble_images,
    generate_output_filepaths,
)

__all__ = ["assemble_images", "generate_output_filepaths"]
