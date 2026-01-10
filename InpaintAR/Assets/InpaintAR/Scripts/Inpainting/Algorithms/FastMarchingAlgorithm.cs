using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    public class FastMarchingAlgorithm : IInpaintingAlgorithm {
        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices) {
            Color[] resultPixels = (Color[])TextureUtility.GetEmptyImagePixels(source).Clone();
            int imageWidth = TextureUtility.GetImageWidth(source);
            int imageHeight = TextureUtility.GetImageHeight(source);
            
            Texture2D resultImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);

            NativeArray<Color32> sourcePixels = source.GetPixelData<Color32>(0);
            Parallel.ForEach(maskPixelIndices, maskPixelIndex => {
                resultPixels[maskPixelIndex] = sourcePixels[maskPixelIndex];
            });
            
            resultImage.SetPixels(resultPixels);
            resultImage.Apply();
            
            return resultImage;
        }
    }
}