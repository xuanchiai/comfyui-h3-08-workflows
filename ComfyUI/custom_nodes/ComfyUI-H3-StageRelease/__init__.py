"""Expose the resource release node used by H3-08."""
from comfy_api.latest import ComfyExtension

from .character_stage import MOSSCharacterStageRelease


class H3StageReleaseExtension(ComfyExtension):
    async def get_node_list(self):
        return [MOSSCharacterStageRelease]


async def comfy_entrypoint():
    return H3StageReleaseExtension()
