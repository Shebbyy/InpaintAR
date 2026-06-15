namespace InpaintAR.Scripts.Inpainting {
    public class CpuInpaintingFactory : IInpaintingFactory {
        public IInpaintingAlgorithm GetInpaintingAlgorithm(InpaintingAlgorithms algorithm) {
            return InpaintingFactory.GetInpaintingAlgorithm(algorithm);
        }
    }
}
