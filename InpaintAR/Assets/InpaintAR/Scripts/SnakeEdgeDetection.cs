using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace InpaintAR.Scripts {
    public static class SnakeEdgeDetection {
        // GVF Snake parameters
        private const int GvfIterations = 20; // Number of iterations for GVF field calculation
        private const int SnakeIterations = 50; // Number of iterations for snake evolution
        private const int SnakeRefinementIterations = 10; // Iterations for refinement when reusing previous contour
        private const float Alpha = 0.05f; // Elasticity (continuity)
        private const float Beta = 0.05f; // Rigidity (curvature)
        private const float Gamma = 0.3f; // Step size for snake evolution
        private const float Mu = 0.2f; // Regularization parameter for GVF

        // Cache for performance optimization
        private static HashSet<int> _cachedRefinedMask;
        private static List<Vector2> _cachedContourPoints;
        private static int _cachedWidth;
        private static int _cachedHeight;

        // Cache for selection mask to avoid recreating HashSet every frame
        private static Rect _cachedSelectionBounds;
        private static List<Vector2> _cachedInitialContour;

        public static HashSet<int> GetContourMaskPixelIndices(RectTransform fillImagePosition,
            Texture2D fillImageTexture,
            RectTransform fillRectMask) {
            CreateSelectionContourForCache(fillImagePosition, fillImageTexture, fillRectMask,
                out Rect maskBounds);

            int width = TextureUtility.GetImageWidth(fillImageTexture);
            int height = TextureUtility.GetImageHeight(fillImageTexture);

            // Existing Snake should be reused
            if (_cachedRefinedMask != null && _cachedContourPoints != null) {
                return RefineWithNewTexture(fillImageTexture, maskBounds);
            }

            _cachedWidth = width;
            _cachedHeight = height;
            _cachedRefinedMask = ApplyGvfSnake(fillImageTexture, maskBounds, true);

            return _cachedRefinedMask;
        }

        public static void ResetSelectionMask() {
            _cachedRefinedMask = null;
            _cachedContourPoints = null;
            _cachedInitialContour = null;
        }

        private static HashSet<int> RefineWithNewTexture(Texture2D fillImageTexture, Rect maskBounds) {
            int width = _cachedWidth;
            int height = _cachedHeight;

            Color[] pixels = fillImageTexture.GetPixels();

            // Compute edge map only in the region of interest (expanded for GVF field)
            int margin = 20; // Margin around mask for GVF computation
            Rect computeRegion = ExpandRect(maskBounds, margin, width, height);

            float[,] edgeMap = ComputeEdgeMapInRegion(pixels, width, height, computeRegion);

            // Compute new GVF field only in the region
            Vector2[,] gvfField = ComputeGvfFieldInRegion(edgeMap, width, height, computeRegion);

            // Refine existing contour with fewer iterations
            List<Vector2> refinedPoints =
                EvolveSnake(_cachedContourPoints, gvfField, width, height, SnakeRefinementIterations);
            _cachedContourPoints = refinedPoints;

            // Convert back to mask
            _cachedRefinedMask = FillContour(refinedPoints, width);

            return _cachedRefinedMask;
        }

        private static void CreateSelectionContourForCache(RectTransform fillImagePosition,
            Texture2D fillImageTexture, RectTransform fillRectMask, out Rect maskBounds) {
            maskBounds = new Rect();

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
            maskBounds = new Rect(pixelLeft, pixelBottom, pixelRight - pixelLeft, pixelTop - pixelBottom);

            if (_cachedSelectionBounds == maskBounds) {
                return;
            }

            _cachedSelectionBounds = maskBounds;
            // Generate initial contour points from rectangular bounds
            // This is much more efficient than ExtractContourFromMask
            _cachedInitialContour = CreateRectangularContour(pixelLeft, pixelRight, pixelBottom, pixelTop);
        }

        private static List<Vector2> CreateRectangularContour(int left, int right, int bottom, int top) {
            // Create a contour by sampling points around the rectangle perimeter
            List<Vector2> contour = new List<Vector2>();
            
            int width = right - left;
            int height = top - bottom;
            int perimeter = 2 * (width + height);
            
            // Sample approximately 100 points around the perimeter
            int targetPoints = Mathf.Min(100, perimeter / 2);
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

        private static HashSet<int> ApplyGvfSnake(Texture2D fillImageTexture, Rect maskBounds,
            bool fullCalculation) {
            int width = TextureUtility.GetImageWidth(fillImageTexture);
            int height = TextureUtility.GetImageHeight(fillImageTexture);

            Color[] pixels = fillImageTexture.GetPixels();

            // Compute edge map only in the region of interest (expanded for GVF field)
            int margin = 20; // Margin around mask for GVF computation
            Rect computeRegion = ExpandRect(maskBounds, margin, width, height);

            // 1. Compute edge map using Sobel operator in region
            float[,] edgeMap = ComputeEdgeMapInRegion(pixels, width, height, computeRegion);

            // 2. Compute GVF field in region
            Vector2[,] gvfField = ComputeGvfFieldInRegion(edgeMap, width, height, computeRegion);

            // 3. Use cached initial contour (generated during mask creation)
            List<Vector2> snakePoints = _cachedInitialContour;

            if (snakePoints == null || snakePoints.Count < 4) return null;

            // 4. Evolve snake
            snakePoints = EvolveSnake(snakePoints, gvfField, width, height, SnakeIterations);

            // Store for visualization/debugging
            if (fullCalculation) {
                _cachedContourPoints = snakePoints;
            }

            // 5. Convert snake points back to mask
            HashSet<int> refinedMask = FillContour(snakePoints, width);

            return refinedMask;
        }

        private static Rect ExpandRect(Rect rect, int margin, int width, int height) {
            return new Rect(
                Mathf.Max(0, rect.x - margin),
                Mathf.Max(0, rect.y - margin),
                Mathf.Min(width - rect.x + margin, rect.width + 2 * margin),
                Mathf.Min(height - rect.y + margin, rect.height + 2 * margin)
            );
        }

        private static float[,] ComputeEdgeMapInRegion(Color[] pixels, int width, int height, Rect region) {
            float[,] edgeMap = new float[width, height];
            float[,] gray = new float[width, height];

            int minX = Mathf.Max(0, (int)region.x);
            int maxX = Mathf.Min(width, (int)(region.x + region.width));
            int minY = Mathf.Max(0, (int)region.y);
            int maxY = Mathf.Min(height, (int)(region.y + region.height));

            // Convert to grayscale only in region
            for (int y = minY; y < maxY; y++) {
                for (int x = minX; x < maxX; x++) {
                    Color c = pixels[y * width + x];
                    gray[x, y] = 0.299f * c.r + 0.587f * c.g + 0.114f * c.b;
                }
            }

            // Sobel edge detection only in region
            for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                    float gx = -gray[x - 1, y - 1] - 2 * gray[x - 1, y] - gray[x - 1, y + 1]
                               + gray[x + 1, y - 1] + 2 * gray[x + 1, y] + gray[x + 1, y + 1];

                    float gy = -gray[x - 1, y - 1] - 2 * gray[x, y - 1] - gray[x + 1, y - 1]
                               + gray[x - 1, y + 1] + 2 * gray[x, y + 1] + gray[x + 1, y + 1];

                    edgeMap[x, y] = Mathf.Sqrt(gx * gx + gy * gy);
                }
            }

            return edgeMap;
        }

        private static Vector2[,] ComputeGvfFieldInRegion(float[,] edgeMap, int width, int height, Rect region) {
            Vector2[,] u = new Vector2[width, height];
            Vector2[,] grad = new Vector2[width, height];

            int minX = Mathf.Max(0, (int)region.x);
            int maxX = Mathf.Min(width, (int)(region.x + region.width));
            int minY = Mathf.Max(0, (int)region.y);
            int maxY = Mathf.Min(height, (int)(region.y + region.height));

            // Compute gradient of edge map only in region
            for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                    float fx = (edgeMap[x + 1, y] - edgeMap[x - 1, y]) * 0.5f;
                    float fy = (edgeMap[x, y + 1] - edgeMap[x, y - 1]) * 0.5f;
                    grad[x, y] = new Vector2(fx, fy);
                    u[x, y] = grad[x, y];
                }
            }

            // Iteratively solve GVF only in region
            for (int iter = 0; iter < GvfIterations; iter++) {
                Vector2[,] uNew = new Vector2[width, height];

                for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                    for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                        // Laplacian of u
                        Vector2 laplacian = (u[x + 1, y] + u[x - 1, y] + u[x, y + 1] + u[x, y - 1] - 4 * u[x, y]);

                        float b = grad[x, y].sqrMagnitude;
                        uNew[x, y] = u[x, y] + Mu * laplacian - b * (u[x, y] - grad[x, y]);
                    }
                }

                // Copy region back
                for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                    for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                        u[x, y] = uNew[x, y];
                    }
                }
            }

            return u;
        }

        private static List<Vector2> EvolveSnake(List<Vector2> points, Vector2[,] gvfField, int width, int height,
            int iterations) {
            int n = points.Count;

            for (int iter = 0; iter < iterations; iter++) {
                List<Vector2> newPoints = new List<Vector2>(n);

                for (int i = 0; i < n; i++) {
                    Vector2 p = points[i];
                    Vector2 prev = points[(i - 1 + n) % n];
                    Vector2 next = points[(i + 1) % n];

                    // Internal forces
                    Vector2 elasticity = Alpha * (prev + next - 2 * p);

                    Vector2 prevPrev = points[(i - 2 + n) % n];
                    Vector2 nextNext = points[(i + 2) % n];
                    Vector2 curvature = Beta * (prevPrev + 2 * prev - 2 * next - nextNext);

                    // External force from GVF
                    int x = Mathf.Clamp((int)p.x, 0, width - 1);
                    int y = Mathf.Clamp((int)p.y, 0, height - 1);
                    Vector2 external = gvfField[x, y];

                    // Update position
                    Vector2 newP = p + Gamma * (elasticity + curvature + external);

                    // Clamp to image bounds
                    newP.x = Mathf.Clamp(newP.x, 0, width - 1);
                    newP.y = Mathf.Clamp(newP.y, 0, height - 1);

                    newPoints.Add(newP);
                }

                points = newPoints;
            }

            return points;
        }

        private static HashSet<int> FillContour(List<Vector2> contour, int width) {
            HashSet<int> mask = new HashSet<int>();
            if (contour.Count < 3) return mask;

            // Find bounding box
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;

            foreach (Vector2 p in contour) {
                minX = Mathf.Min(minX, p.x);
                maxX = Mathf.Max(maxX, p.x);
                minY = Mathf.Min(minY, p.y);
                maxY = Mathf.Max(maxY, p.y);
            }

            ScanlineFillMask(mask, contour, width, (int)minY, (int)maxY, (int)minX, (int)maxX);
            
            
            return mask;
        }

        /**
         * Polygon Filling Algorithm
         * Process each scanline and find edge intersections
         */
        private static void ScanlineFillMask(HashSet<int> mask, List<Vector2> contour, int width, int yMin, int yMax, int xMin,
            int xMax) {

            Parallel.For(yMin, yMax + 1, () => new List<int>(), (y, _, localList) => {
                List<float> intersections = new List<float>(contour.Count);
                
                FillEdgeIntersectionList(intersections, contour, y);
                
                // Sort intersections and fill between pairs of intersections
                intersections.Sort();

                for (int i = 0; i < intersections.Count - 1; i += 2) {
                    int xStart = Mathf.Max(xMin, Mathf.CeilToInt(intersections[i]));
                    int xEnd = Mathf.Min(xMax, Mathf.FloorToInt(intersections[i + 1]));

                    for (int x = xStart; x <= xEnd; x++) {
                        localList.Add(y * width + x);
                    }
                }

                return localList;
            }, res => {
                lock (mask) {
                    mask.AddRange(res);
                }
            });
        }

        private static void FillEdgeIntersectionList(List<float> intersections, List<Vector2> contour, int y) {
            intersections.Clear();
            float scanY = y + 0.5f; // Center of pixel

            int n = contour.Count;
            for (int i = 0; i < n; i++) {
                Vector2 p1 = contour[i];
                Vector2 p2 = contour[(i + 1) % n];

                float y1 = p1.y;
                float y2 = p2.y;

                // whilst == would be preferrable, rounding to avoid floating point errors would make it less performant than this
                if (   (y1 < scanY && y2 < scanY) 
                    || (y1 > scanY && y2 > scanY)) continue;
                
                // Intersection calculation
                float t = (scanY - y1) / (y2 - y1);
                float x = p1.x + t * (p2.x - p1.x);
                intersections.Add(x);
            }
        }
    }
}