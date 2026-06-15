using System;
using System.Collections.Generic;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {
    // Bridges the CPU-style IInpaintingAlgorithm contract the consumer uses to a GPU-resident
    // IGpuInpaintingAlgorithm. Blits the (already GPU-resident) source into a cached RenderTexture -
    // a GPU->GPU copy, no CPU readback - and reduces the mask index set to its bounding RectInt. Both
    // conversions are lossless for this app, since the mask produced by MaskController is always a
    // filled rectangle. The blit also gives us a stable ARGB32 copy decoupled from the volatile
    // passthrough texture (which the async benchmarking read-back samples a frame later).
    //
    // Ownership note: the returned RenderTexture is owned by the wrapped GPU algorithm (reused every
    // frame), so the consumer must NOT destroy it - it is released here in Dispose().
    public class GpuInpaintingAdapter : IInpaintingAlgorithm, IDisposable {
        private readonly IGpuInpaintingAlgorithm m_gpu;
        private RenderTexture m_sourceRT;
        private int m_width, m_height;

        public GpuInpaintingAdapter(IGpuInpaintingAlgorithm gpu) {
            m_gpu = gpu;
        }

        public Texture Inpaint(Texture source, HashSet<int> maskPixelIndices) {
            if (source == null) return null;

            EnsureSourceRT(source.width, source.height);
            Graphics.Blit(source, m_sourceRT);

            RectInt bounds = ComputeBounds(maskPixelIndices, source.width);
            return m_gpu.Inpaint(m_sourceRT, bounds);
        }

        private void EnsureSourceRT(int width, int height) {
            if (m_sourceRT != null && m_width == width && m_height == height) return;
            if (m_sourceRT != null) m_sourceRT.Release();

            m_width = width;
            m_height = height;
            m_sourceRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
            m_sourceRT.Create();
        }

        // Bounding box of the (rectangular) mask in full-resolution pixel space.
        private static RectInt ComputeBounds(HashSet<int> maskPixelIndices, int width) {
            if (maskPixelIndices == null || maskPixelIndices.Count == 0) {
                return new RectInt(0, 0, 0, 0);
            }

            int minX = int.MaxValue, minY = int.MaxValue, maxX = int.MinValue, maxY = int.MinValue;
            foreach (int index in maskPixelIndices) {
                int x = index % width;
                int y = index / width;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
            return new RectInt(minX, minY, maxX - minX + 1, maxY - minY + 1);
        }

        public void Dispose() {
            if (m_sourceRT != null) {
                m_sourceRT.Release();
                m_sourceRT = null;
            }
            m_gpu.Dispose();
        }
    }
}
