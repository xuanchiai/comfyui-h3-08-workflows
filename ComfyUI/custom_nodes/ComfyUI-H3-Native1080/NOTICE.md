# Experimental native RGB streaming delivery

Code is GPL-3.0, following the local `ComfyUI-H3-X2-Delivery/streaming.py` adapter and its upstream [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream/tree/eb148a3d0f91a273668bbe027bfcbafb85f2c5e8). See the copied LICENSE.

The candidate uses ComfyUI's original H3 decoder, temporal blending and spatial tiling. It removes the X2 packed RGB path. No X2 weights or model weights are included. This code license does not replace the model's applicable license.

This is a CPU-checked experimental delivery adapter, not a GPU-validated workflow or a quality-approved production node.
