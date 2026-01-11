using System.Collections.Generic;
using InpaintAR.Scripts.Util;
using UnityEngine;

namespace InpaintAR.Scripts.SnakeEdgeDetection {
    public static class SnakeController {
        private const int SnakeIterations = 15000; // Number of iterations for initialization
        private const int SnakeRefinementIterations = 2000; // Iterations for refinement per frame (more -> quicker adjustment to movements)
        private const int InitialPerimeterPointCount = 200; // Amount of Points for the contour
        
        private static int _cachedWidth;
        private static int _cachedHeight;
        private static Rect _cachedSelectionBounds;
        private static readonly BalloonSnake Snake = new();

        public static HashSet<int> GetContourMaskPixelIndices(RectTransform fillImagePosition,
            Texture2D fillImageTexture,
            RectTransform fillRectMask) {
            CreateSelectionContourForCache(fillImagePosition, fillImageTexture, fillRectMask);

            int width = TextureUtility.GetImageWidth(fillImageTexture);
            int height = TextureUtility.GetImageHeight(fillImageTexture);

            // Existing Snake should be reused
            int iterations;
            if (Snake.GetRefinedMask() != null && Snake.GetContourPoints() != null) {
                iterations = SnakeRefinementIterations;
            }
            else {
                _cachedWidth = width;
                _cachedHeight = height;

                Snake.InitializeCacheVariables(_cachedWidth, _cachedHeight);
                iterations = SnakeIterations;
            }
            
            return Snake.ApplyBalloonSnake(fillImageTexture, _cachedSelectionBounds, iterations, _cachedWidth, _cachedHeight);
        }

        public static void ResetSelectionMask() {
            Snake.ResetSelectionMask();
            
        }

        private static void CreateSelectionContourForCache(RectTransform fillImagePosition,
            Texture2D fillImageTexture, RectTransform fillRectMask) {

            if (!fillImageTexture || !fillRectMask) {
                return;
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

            // Store mask bounds in pixel coordinates
            var maskBounds = new Rect(pixelLeft, pixelBottom, pixelRight - pixelLeft, pixelTop - pixelBottom);

            if (_cachedSelectionBounds == maskBounds) {
                return;
            }

            var contourPoints = Snake.GetContourPoints();
            if (contourPoints != null) {
                Vector2 offset = new Vector2(
                    maskBounds.x - _cachedSelectionBounds.x,
                    maskBounds.y - _cachedSelectionBounds.y
                );
                
                TranslateContourPoints(contourPoints, offset, imageWidth, imageHeight);
            }
            else {
                Snake.SetContourPoints(CreateRectangularContour(pixelLeft, pixelRight, pixelBottom, pixelTop));
            }

            _cachedSelectionBounds = maskBounds;
        }

        private static List<Vector2> CreateRectangularContour(int left, int right, int bottom, int top) {
            List<Vector2> contour = new List<Vector2>();

            int width = right - left;
            int height = top - bottom;
            int perimeter = 2 * (width + height);

            int targetPoints = Mathf.Max(InitialPerimeterPointCount, perimeter / 2);
            float step = perimeter / (float)targetPoints;

            for (int i = 0; i < targetPoints; i++) {
                float t = i * step;
                Vector2 point;

                if (t < width) {
                    point = new Vector2(left + t, bottom);
                }
                else if (t < width + height) {
                    point = new Vector2(right - 1, bottom + (t - width));
                }
                else if (t < 2 * width + height) {
                    point = new Vector2(right - 1 - (t - width - height), top - 1);
                }
                else {
                    point = new Vector2(left, top - 1 - (t - 2 * width - height));
                }

                contour.Add(point);
            }

            return contour;
        }

        private static void TranslateContourPoints(List<Vector2> points, Vector2 offset, int width, int height) {
            for (int i = 0; i < points.Count; i++) {
                Vector2 translatedPoint = points[i] + offset;
                
                // Clamp to image bounds
                translatedPoint.x = Mathf.Clamp(translatedPoint.x, 0, width - 1);
                translatedPoint.y = Mathf.Clamp(translatedPoint.y, 0, height - 1);
                
                points[i] = translatedPoint;
            }
        }
    }
}