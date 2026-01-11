using System.Collections.Generic;
using System.Threading.Tasks;
using InpaintAR.Scripts.Util;
using Unity.Collections;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    public class FastMarchingAlgorithm : IInpaintingAlgorithm {
        private Color[] m_pixelBuffer;
        public Texture2D Inpaint(Texture2D source, HashSet<int> maskPixelIndices) {
            int imageWidth = TextureUtility.GetImageWidth(source);
            int imageHeight = TextureUtility.GetImageHeight(source);
            int pixelCount = imageWidth * imageHeight;
            
            if (m_pixelBuffer == null || m_pixelBuffer.Length != pixelCount) {
                m_pixelBuffer = new Color[pixelCount];
            }
            System.Array.Copy(TextureUtility.GetEmptyImagePixels(source), m_pixelBuffer, pixelCount);
            
            Texture2D resultImage = new Texture2D(imageWidth, imageHeight, TextureFormat.RGBA32, false);

            NativeArray<Color32> sourcePixels = source.GetPixelData<Color32>(0);
            Parallel.ForEach(maskPixelIndices, maskPixelIndex => {
                m_pixelBuffer[maskPixelIndex] = sourcePixels[maskPixelIndex];
            });
            
            resultImage.SetPixels(m_pixelBuffer);
            resultImage.Apply();
            
            return resultImage;
        }
    }
}