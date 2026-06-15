using System;
using InpaintAR.Scripts.Inpainting.Algorithms;

namespace InpaintAR.Scripts.Inpainting {
    // GPU path. Maps the same InpaintingAlgorithms enum to the GPU-resident compute / Sentis
    // implementations, each wrapped in GpuInpaintingAdapter so it satisfies IInpaintingAlgorithm.
    // Drop-in replacement for CpuInpaintingFactory.
    public class InpaintComputeFactory : IInpaintingFactory {
        public IInpaintingAlgorithm GetInpaintingAlgorithm(InpaintingAlgorithms algorithm) {
            IGpuInpaintingAlgorithm gpu = algorithm switch {
                InpaintingAlgorithms.FastMarchingMethod => new FastMarchingComputeAlgorithm(),
                InpaintingAlgorithms.ExemplarLocalTextureMatching => new EltmComputeAlgorithm(),
                InpaintingAlgorithms.NonLinearTextureMatching => new NltmComputeAlgorithm(),
                InpaintingAlgorithms.LargeMaskInpaintingInference => new LaMaComputeAlgorithm(),
                _ => throw new NotImplementedException($"GPU mapping for Algorithm {algorithm} is missing!")
            };
            return new GpuInpaintingAdapter(gpu);
        }
    }
}
