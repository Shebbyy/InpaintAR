using System.Collections.Generic;
using System.Diagnostics;
using InpaintAR.Scripts.Benchmarking.Evaluators;
using InpaintAR.Scripts.Util;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {

    public enum InpaintingAlgorithms {
        FastMarchingMethod,
        NonLinearTextureMatching,
        ExemplarLocalTextureMatching,
        LargeMaskInpaintingInference 
    }
    
    public abstract class AbstractInpaintingAlgorithm : IInpaintingAlgorithm {
        private readonly Stopwatch m_watch = new();
        protected Color32[] PixelBuffer;
        private Color32[] m_originalPixelBuffer;

        public Texture Inpaint(Texture source, HashSet<int> maskPixelIndices) {
            // CPU algorithms operate on CPU-readable pixels, so the (usually GPU) source is pulled
            // down to a Texture2D here. This is the single GPU->CPU readback on the CPU path; the GPU
            // path (GpuInpaintingAdapter) never does it. Kept outside the stopwatch so the recorded
            // inpainting time stays comparable to before this readback was moved in from the caller.
            Texture2D sourceTex = source as Texture2D ?? TextureUtility.CopyTexture(source);
            bool ownsCopy = source is not Texture2D;

            m_watch.Reset();
            m_watch.Start();

            // Save original pixels before inpainting for quality evaluation
            m_originalPixelBuffer = sourceTex.GetPixels32();
            PixelBuffer = sourceTex.GetPixels32();

            var texture = InpaintLogic(sourceTex, maskPixelIndices);

            m_watch.Stop();
            PerformanceEvaluator.AddInpaintingStats(maskPixelIndices.Count, m_watch.ElapsedMilliseconds);
            QualityEvaluator.EvaluateQuality(m_originalPixelBuffer, PixelBuffer, sourceTex.width, sourceTex.height, maskPixelIndices);
            ClutterEvaluator.EvaluateClutterReduction(m_originalPixelBuffer, PixelBuffer, sourceTex.width, sourceTex.height, maskPixelIndices);

            if (ownsCopy) Object.Destroy(sourceTex);
            return texture;
        }

        protected abstract Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices);
    }
}