using System.Collections.Generic;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    public class LaMaAlgorithm : IInpaintingAlgorithm {
        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices) {
            return source;
        }
    }
}