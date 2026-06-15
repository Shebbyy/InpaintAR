using System;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {
    // GPU-resident inpainting contract. Unlike AbstractInpaintingAlgorithm (which round-trips
    // through Color32[] on the CPU), the source stays on the GPU and the result is a
    // RenderTexture, so nothing blocks the main thread in the hot path.
    public interface IGpuInpaintingAlgorithm : IDisposable {
        // maskBounds is in full-resolution pixel space (x=left, y=bottom, width, height).
        RenderTexture Inpaint(RenderTexture source, RectInt maskBounds);
    }
}
