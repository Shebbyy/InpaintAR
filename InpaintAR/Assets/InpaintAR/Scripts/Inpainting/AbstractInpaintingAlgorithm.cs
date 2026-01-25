using System.Collections.Generic;
using System.Diagnostics;
using InpaintAR.Scripts.Benchmarking.Evaluators;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {

    public enum InpaintingAlgorithms {
        FastMarchingMethod,
        NonLinearTextureMatching,
        ExemplarLocalTextureMatching,
        LargeMaskInpaintingInference 
    }
    
    public abstract class AbstractInpaintingAlgorithm {
        private readonly Stopwatch m_watch = new();
        protected Color32[] PixelBuffer;
        private Color32[] m_originalPixelBuffer;

        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices) {
            m_watch.Reset();
            m_watch.Start();

            // Save original pixels before inpainting for quality evaluation
            m_originalPixelBuffer = source.GetPixels32();
            PixelBuffer = source.GetPixels32();

            var texture = InpaintLogic(source, maskPixelIndices);

            m_watch.Stop();
            PerformanceEvaluator.AddInpaintingStats(maskPixelIndices.Count, m_watch.ElapsedMilliseconds);
            QualityEvaluator.EvaluateQuality(m_originalPixelBuffer, PixelBuffer, source.width, source.height, maskPixelIndices);
            ClutterEvaluator.EvaluateClutterReduction(m_originalPixelBuffer, PixelBuffer, source.width, source.height, maskPixelIndices);

            return texture;
        }

        protected abstract Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices);
    }
}