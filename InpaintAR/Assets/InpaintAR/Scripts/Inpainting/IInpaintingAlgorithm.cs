using System.Collections.Generic;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {

    public enum InpaintingAlgorithms {
        FastMarchingMethod,
        NonLinearTextureMatching,
        ExemplarLocalTextureMatching,
        LargeMaskInpaintingInference
    }
    
    public interface IInpaintingAlgorithm {
        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices);
    }
}