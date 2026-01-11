using System;
using InpaintAR.Scripts.Inpainting.Algorithms;

namespace InpaintAR.Scripts.Inpainting {
    
    public static class InpaintingFactory {
        public static AbstractInpaintingAlgorithm GetInpaintingAlgorithm(InpaintingAlgorithms algorithm) {
            return algorithm switch {
                InpaintingAlgorithms.FastMarchingMethod => new FastMarchingAlgorithm(),
                InpaintingAlgorithms.ExemplarLocalTextureMatching => new EltmAlgorithm(),
                InpaintingAlgorithms.NonLinearTextureMatching => new NltmAlgorithm(),
                InpaintingAlgorithms.LargeMaskInpaintingInference => new LaMaAlgorithm(),
                _ => throw new NotImplementedException($"Mapping for Algorithm {algorithm} is missing!")
            };
        }
    }
}