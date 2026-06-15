using System.Collections.Generic;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {
    // Unified runtime contract the consumer (AreaSelectionVisualizer) talks to, so the CPU
    // (AbstractInpaintingAlgorithm) and GPU (AbstractGpuInpaintingAlgorithm via GpuInpaintingAdapter)
    // implementations are interchangeable.
    //
    // The source is passed as the live (GPU) Texture: the CPU path reads it back to a Texture2D
    // internally (only when it actually needs CPU pixels), while the GPU path keeps it on the GPU.
    // Returns Texture - the common base of both Texture2D (CPU result) and RenderTexture (GPU result).
    public interface IInpaintingAlgorithm {
        Texture Inpaint(Texture source, HashSet<int> maskPixelIndices);
    }
}
