namespace InpaintAR.Scripts.Inpainting {
    // Swappable factory contract. CpuInpaintingFactory (Burst CPU) and InpaintComputeFactory
    // (GPU compute / Sentis) are interchangeable behind this interface, so the consumer can be
    // hot-swapped between the two without structural changes.
    public interface IInpaintingFactory {
        IInpaintingAlgorithm GetInpaintingAlgorithm(InpaintingAlgorithms algorithm);
    }
}
