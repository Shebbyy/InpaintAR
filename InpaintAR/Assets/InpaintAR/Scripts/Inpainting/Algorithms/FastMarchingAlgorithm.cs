using System.Collections.Generic;
using System.Threading.Tasks;
using InpaintAR.Scripts.Util;
using Unity.Collections;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    public class FastMarchingAlgorithm : AbstractInpaintingAlgorithm {
        
        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            int imageWidth = TextureUtility.GetImageWidth(source);
            int imageHeight = TextureUtility.GetImageHeight(source);
            int pixelCount = imageWidth * imageHeight;
            
            if (m_inpaintedPixelBuffer == null || m_inpaintedPixelBuffer.Length != pixelCount) {
                m_inpaintedPixelBuffer = new Color32[pixelCount];
            }
            System.Array.Copy(TextureUtility.GetEmptyImagePixels(source), m_inpaintedPixelBuffer, pixelCount);
            
            Texture2D resultImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);

            NativeArray<Color32> sourcePixels = source.GetPixelData<Color32>(0);
            m_sourcePixelBuffer = source.GetPixels32();
            
            Parallel.ForEach(maskPixelIndices, maskPixelIndex => {
                m_inpaintedPixelBuffer[maskPixelIndex] = sourcePixels[maskPixelIndex];
            });
            
            resultImage.SetPixels32(m_inpaintedPixelBuffer);
            resultImage.Apply();
            
            return resultImage;
        }
    }
}