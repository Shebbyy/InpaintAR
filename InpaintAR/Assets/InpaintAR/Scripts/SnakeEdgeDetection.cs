using System.Collections.Generic;
using UnityEngine;

namespace InpaintAR.Scripts {
    public static class SnakeEdgeDetection {
        public static HashSet<int> GetContourMaskPixelIndices(RectTransform fillImagePosition, Texture fillImageTexture, RectTransform fillRectMask) {
            var selectionMask = GetSelectionMaskPixelIndices(fillImagePosition, fillImageTexture, fillRectMask);

            return selectionMask;
        }

        private static HashSet<int> GetSelectionMaskPixelIndices(RectTransform fillImagePosition, Texture fillImageTexture, RectTransform fillRectMask) {
            if (!fillImageTexture || !fillRectMask) {
                return null;
            }

            int imageWidth = TextureUtility.GetImageWidth(fillImageTexture);
            int imageHeight = TextureUtility.GetImageHeight(fillImageTexture);
            
            // Get the mask rectangle bounds in local coordinates relative to fillImagePosition
            Vector2 maskLocalPos = fillRectMask.localPosition;
            Vector2 maskSize = fillRectMask.sizeDelta;

            Vector2 imageLocalPos = fillImagePosition.localPosition;
            Vector2 imageSize = fillImagePosition.sizeDelta;

            Vector2 maskOffset = maskLocalPos - imageLocalPos;

            // Conversion World Coordinates -> Texture Coordinates (0-1)
            float normLeft = maskOffset.x / imageSize.x;
            float normBottom = maskOffset.y / imageSize.y;
            float normRight = (maskOffset.x + maskSize.x) / imageSize.x;
            float normTop = (maskOffset.y + maskSize.y) / imageSize.y;

            int pixelLeft = Mathf.Max(0, Mathf.FloorToInt(normLeft * imageWidth));
            int pixelRight = Mathf.Min(imageWidth, Mathf.CeilToInt(normRight * imageWidth));
            int pixelBottom = Mathf.Max(0, Mathf.FloorToInt(normBottom * imageHeight));
            int pixelTop = Mathf.Min(imageHeight, Mathf.CeilToInt(normTop * imageHeight));

            // Collect only the pixel coordinates that are in the mask
            HashSet<int> maskPixels = new HashSet<int>();
            for (int y = pixelBottom; y < pixelTop; y++) {
                for (int x = pixelLeft; x < pixelRight; x++) {
                    maskPixels.Add(imageWidth * y + x);
                }
            }

            return maskPixels;
        }

        
    }
}