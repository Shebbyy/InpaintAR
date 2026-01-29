using System.Collections.Generic;
using InpaintAR.Scripts.Util;
using UnityEngine;

namespace InpaintAR.Scripts.MaskCreation {
    public static class MaskController {
        private static Rect _cachedSelectionBounds;
        private static HashSet<int> _cachedMask;

        public static HashSet<int> GetMaskPixelIndices(RectTransform fillImagePosition,
            Texture2D fillImageTexture,
            RectTransform fillRectMask) {

            if (!fillImageTexture || !fillRectMask) {
                return _cachedMask ?? new HashSet<int>();
            }

            int imageWidth = TextureUtility.GetImageWidth(fillImageTexture);
            int imageHeight = TextureUtility.GetImageHeight(fillImageTexture);

            var imageSize = fillImagePosition.sizeDelta;

            Vector2 maskLocalPosInImage = -fillImagePosition.localPosition;
            Vector2 maskSize = fillRectMask.sizeDelta;

            // Conversion Local Canvas Coordinates -> Normalized Texture Coordinates (0-1)
            float normLeft = maskLocalPosInImage.x / imageSize.x;
            float normBottom = maskLocalPosInImage.y / imageSize.y;
            float normRight = (maskLocalPosInImage.x + maskSize.x) / imageSize.x;
            float normTop = (maskLocalPosInImage.y + maskSize.y) / imageSize.y;

            int pixelLeft = Mathf.Max(0, Mathf.FloorToInt(normLeft * imageWidth));
            int pixelRight = Mathf.Min(imageWidth, Mathf.CeilToInt(normRight * imageWidth));
            int pixelBottom = Mathf.Max(0, Mathf.FloorToInt(normBottom * imageHeight));
            int pixelTop = Mathf.Min(imageHeight, Mathf.CeilToInt(normTop * imageHeight));

            var maskBounds = new Rect(pixelLeft, pixelBottom, pixelRight - pixelLeft, pixelTop - pixelBottom);

            // Skip if bounds haven't changed
            if (_cachedSelectionBounds == maskBounds && _cachedMask != null) {
                return _cachedMask;
            }

            _cachedSelectionBounds = maskBounds;

            // Fill rectangle directly
            int estimatedSize = (pixelRight - pixelLeft) * (pixelTop - pixelBottom);
            _cachedMask = new HashSet<int>(estimatedSize);

            for (int y = pixelBottom; y < pixelTop; y++) {
                int rowOffset = y * imageWidth;
                for (int x = pixelLeft; x < pixelRight; x++) {
                    _cachedMask.Add(rowOffset + x);
                }
            }

            return _cachedMask;
        }

        public static void ResetSelectionMask() {
            _cachedMask = null;
        }
    }
}
