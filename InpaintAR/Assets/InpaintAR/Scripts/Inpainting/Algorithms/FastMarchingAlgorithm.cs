using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    public class FastMarchingAlgorithm : IInpaintingAlgorithm {
        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices) {
            Color[] resultPixels = (Color[])TextureUtility.GetEmptyImagePixels(source).Clone();
            int imageWidth = TextureUtility.GetImageWidth(source);
            int imageHeight = TextureUtility.GetImageHeight(source);
            
            Texture2D resultImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);

            Color[] sourcePixels = source.GetPixels();
            Parallel.ForEach(maskPixelIndices, maskPixelIndex => {
                resultPixels[maskPixelIndex] = sourcePixels[maskPixelIndex];
            });
            
            resultImage.SetPixels(resultPixels);
            resultImage.Apply();
            
            return resultImage;
        }
    }
}