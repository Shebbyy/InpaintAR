using UnityEngine;

namespace InpaintAR.Scripts.Inpainting {

    public enum InpaintingAlgorithms {
        FastMarchingMethod,
        NonLinearTextureMatching,
        ExemplarLocalTextureMatching,
        LargeMaskInpaintingInference
    }
    
    public interface IInpaintingAlgorithm {
        public Texture2D Inpaint(Texture2D source, Texture2D mask);
    }
}