using System;

namespace InpaintAR.Scripts.Inpainting {
    
    public static class InpaintingFactory {
        public static IInpaintingAlgorithm GetInpaintingAlgorithm(InpaintingAlgorithms algorithm) {
            switch (algorithm) {
                default:
                    throw new NotImplementedException($"Mapping for Algorithm {algorithm} is missing!");
            }
        }
    }
}