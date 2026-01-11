using System.Collections.Generic;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    public class EltmAlgorithm : AbstractInpaintingAlgorithm {
        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            return source;
        }
    }
}