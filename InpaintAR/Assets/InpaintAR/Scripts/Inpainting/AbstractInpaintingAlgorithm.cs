using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {

    public enum InpaintingAlgorithms {
        FastMarchingMethod,
        NonLinearTextureMatching,
        ExemplarLocalTextureMatching,
        LargeMaskInpaintingInference
    }
    
    public abstract class AbstractInpaintingAlgorithm {
        protected readonly Stopwatch Watch = new();
        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices, out long elapsedTime) {
            Watch.Reset();
            Watch.Start();
            
            var texture = InpaintLogic(source, maskPixelIndices);
            
            Watch.Stop();
            elapsedTime = Watch.ElapsedMilliseconds;

            return texture;
        }

        protected abstract Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices);
    }
}