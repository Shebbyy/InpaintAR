using System.Collections.Generic;
using System.Diagnostics;
using InpaintAR.Scripts.Benchmarking;
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
        protected Color32[] MInpaintedPixelBuffer;
        protected Color32[] MSourcePixelBuffer;
        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices) {
            m_watch.Reset();
            m_watch.Start();
            
            var texture = InpaintLogic(source, maskPixelIndices);
            
            m_watch.Stop();
            PerformanceEvaluator.AddInpaintingStats(maskPixelIndices.Count, m_watch.ElapsedMilliseconds);
            QualityEvaluator.EvaluateQuality(MSourcePixelBuffer, MInpaintedPixelBuffer, source.width, source.height, maskPixelIndices);
            
            return texture;
        }

        protected abstract Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices);
    }
}