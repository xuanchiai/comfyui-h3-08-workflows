"""Return a stage output unchanged and release weights the next stage will not use."""
from comfy import model_management
from comfy_api.latest import IO

from .resident_pages import release_resident_pages


class MOSSCharacterStageRelease(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MOSSCharacterStageRelease",
            display_name="MOSS 人物阶段资源释放",
            category="MOSS/资源",
            description="等待本阶段完成，原样传递结果，释放指定的编码器、模型或 VAE。",
            inputs=[
                IO.AnyType.Input("value", display_name="阶段结果"),
                IO.AnyType.Input("after", optional=True, display_name="同时等待"),
                IO.Clip.Input("clip", optional=True, display_name="释放编码器"),
                IO.Model.Input("model", optional=True, display_name="释放模型"),
                IO.Vae.Input("vae", optional=True, display_name="释放 VAE"),
                IO.Boolean.Input("reclaim_ram", default=True, display_name="回收驻留 RAM"),
            ],
            outputs=[IO.AnyType.Output(display_name="原结果")],
        )

    @classmethod
    def execute(cls, value, reclaim_ram=True, after=None, clip=None, model=None, vae=None):
        patchers = tuple(patcher for patcher in (
            clip.patcher if clip is not None else None,
            model,
            vae.patcher if vae is not None else None,
        ) if patcher is not None)
        for patcher in patchers:
            model_management.unload_model_and_clones(patcher)
        if patchers:
            model_management.reset_cast_buffers()
        if reclaim_ram:
            release_resident_pages(768)
        return IO.NodeOutput(value)
