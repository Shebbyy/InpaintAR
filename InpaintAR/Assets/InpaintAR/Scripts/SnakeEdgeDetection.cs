using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;

namespace InpaintAR.Scripts {
    public static class SnakeEdgeDetection {
        // Thread-local pool for intersection lists to avoid allocations
        private static readonly ThreadLocal<List<float>> IntersectionPool = new(() => new List<float>(100));
        
        // Profiler markers for key operations
        private static readonly ProfilerMarker SEdgeMapMarker = new("SnakeEdge.ComputeEdgeMap");
        private static readonly ProfilerMarker SEvolveSnakeMarker = new("SnakeEdge.EvolveSnake");
        private static readonly ProfilerMarker SFillContourMarker = new("SnakeEdge.FillContour");
        
        private const int SnakeIterations = 15000; // Number of iterations for snake evolution
        private const int SnakeRefinementIterations = 3000; // Iterations for refinement in subsequent frames
        private const float Elasticity = 2.5f; // increase for smoother contours
        private const float Rigidity = 1f; // increase to prevent breaking apart
        private const float PositionScaling = 0.12f; // How much the Position change gets scaled in total
        private const float MovementPerFrame = 6f; // movement per frame
        private const float EdgeAttraction = 125.0f; // increased edge attraction
        private const float EdgeThreshold = 0.025f; // Threshold for edge detection - lower to detect more edges
        private const int InitialPerimeterPointCount = 200; // Reduced point count for more stable evolution
        
        private const float BarrierWeight = 100.0f; // Weight of the counter power when close
        private const float StabilizationThreshold = 0.012f; // Average movement threshold for early stopping

        private static HashSet<int> _cachedRefinedMask;
        private static List<Vector2> _cachedContourPoints;
        private static int _cachedWidth;
        private static int _cachedHeight;
        private static Rect _cachedSelectionBounds;

        public static HashSet<int> GetContourMaskPixelIndices(RectTransform fillImagePosition,
            Texture2D fillImageTexture,
            RectTransform fillRectMask) {
            CreateSelectionContourForCache(fillImagePosition, fillImageTexture, fillRectMask);

            int width = TextureUtility.GetImageWidth(fillImageTexture);
            int height = TextureUtility.GetImageHeight(fillImageTexture);

            // Existing Snake should be reused
            if (_cachedRefinedMask != null && _cachedContourPoints != null) {
                ApplyBalloonSnake(fillImageTexture, _cachedSelectionBounds, SnakeRefinementIterations);
            }
            else {
                _cachedWidth = width;
                _cachedHeight = height;
                ApplyBalloonSnake(fillImageTexture, _cachedSelectionBounds, SnakeIterations);
            }

            

            return _cachedRefinedMask;
        }

        public static void ResetSelectionMask() {
            _cachedRefinedMask = null;
            _cachedContourPoints = null;
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

            if (_cachedContourPoints != null) {
                Vector2 offset = new Vector2(
                    maskBounds.x - _cachedSelectionBounds.x,
                    maskBounds.y - _cachedSelectionBounds.y
                );
                
                TranslateContourPoints(_cachedContourPoints, offset, imageWidth, imageHeight);
            }
            else {
                _cachedContourPoints = CreateRectangularContour(pixelLeft, pixelRight, pixelBottom, pixelTop);
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

        private static void ApplyBalloonSnake(Texture2D fillImageTexture, Rect maskBounds, int iterations) {
            int width = _cachedWidth;
            int height = _cachedHeight;

            NativeArray<Color32> pixels = fillImageTexture.GetPixelData<Color32>(0);

            int margin = 20;
            Rect computeRegion = ExpandRect(maskBounds, margin, width, height);

            float[,] edgeMap = ComputeSobelEdgeMapInRegion(pixels, width, height, computeRegion);
            
            var snakePoints = EvolveBalloonSnake(_cachedContourPoints, edgeMap, width, height, iterations);

            _cachedContourPoints = snakePoints;

            _cachedRefinedMask = FillContour(snakePoints, width);
        }

        private static Rect ExpandRect(Rect rect, int margin, int width, int height) {
            return new Rect(
                Mathf.Max(0, rect.x - margin),
                Mathf.Max(0, rect.y - margin),
                Mathf.Min(width - rect.x + margin, rect.width + 2 * margin),
                Mathf.Min(height - rect.y + margin, rect.height + 2 * margin)
            );
        }

        private static float[,] ComputeSobelEdgeMapInRegion(NativeArray<Color32> pixels, int width, int height, Rect region) {
            SEdgeMapMarker.Begin();
            
            float[,] edgeMap = new float[width, height];
            float[][] gray = new float[width][];
            for (int index = 0; index < width; index++) {
                gray[index] = new float[height];
            }

            int minX = Mathf.Max(0, (int)region.x);
            int maxX = Mathf.Min(width, (int)(region.x + region.width));
            int minY = Mathf.Max(0, (int)region.y);
            int maxY = Mathf.Min(height, (int)(region.y + region.height));

            // Convert to grayscale
            for (int y = minY; y < maxY; y++) {
                for (int x = minX; x < maxX; x++) {
                    Color c = pixels[y * width + x];
                    gray[x][y] = 0.299f * c.r + 0.587f * c.g + 0.114f * c.b;
                }
            }

            float maxEdgeEnergy = 0f;

            // Sobel: compute squared gradient magnitude |∇I|²
            for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                    float gx =
                        -gray[x - 1][y - 1] - 2f * gray[x - 1][y] - gray[x - 1][y + 1] +
                        gray[x + 1][y - 1] + 2f * gray[x + 1][y] + gray[x + 1][y + 1];

                    float gy =
                        -gray[x - 1][y - 1] - 2f * gray[x][y - 1] - gray[x + 1][y - 1] +
                        gray[x - 1][y + 1] + 2f * gray[x][y + 1] + gray[x + 1][y + 1];
                    
                    float gradMag2 = gx * gx + gy * gy;

                    edgeMap[x, y] = gradMag2;
                    if (gradMag2 > maxEdgeEnergy) {
                        maxEdgeEnergy = gradMag2;
                    }
                }
            }

            if (maxEdgeEnergy <= 0f) return edgeMap;

            float invMax = 1f / maxEdgeEnergy;

            // normalization
            for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                    float normalized = edgeMap[x, y] * invMax;

                    // Threshold weak edges
                    edgeMap[x, y] = normalized >= EdgeThreshold ? normalized : 0f;
                }
            }

            SEdgeMapMarker.End();
            return edgeMap;
        }

        private static List<Vector2> EvolveBalloonSnake(List<Vector2> points, float[,] edgeMap, int width, int height,
            int iterations) {
            SEvolveSnakeMarker.Begin();
            
            int n = points.Count;

            // Gradient Gaussian Field for stability
            Vector2[,] gradientField = ComputeGradientField(edgeMap, width, height);

            for (int iter = 0; iter < iterations; iter++) {
                List<Vector2> newPoints = new List<Vector2>(n);
                float totalMovement = 0f;

                for (int i = 0; i < n; i++) {
                    Vector2 p = points[i];
                    Vector2 prev = points[(i - 1 + n) % n];
                    Vector2 next = points[(i + 1) % n];

                    // Internal forces (smoothness constraints)
                    Vector2 elasticity = Elasticity * (prev + next - 2 * p);

                    Vector2 prevPrev = points[(i - 2 + n) % n];
                    Vector2 nextNext = points[(i + 2) % n];
                    Vector2 curvature = Rigidity * (prevPrev - 2 * prev + 2 * next - nextNext);

                    Vector2 tangent = (next - prev).normalized;
                    var newX = -tangent.y;
                    var newY = tangent.x;
                    Vector2 normal = new Vector2(newX, newY); // 90° rotation

                    // External forces from image
                    int x = Mathf.Clamp((int)p.x, 0, width - 1);
                    int y = Mathf.Clamp((int)p.y, 0, height - 1);

                    Vector2 grad = gradientField[x, y];
                    float edgeStrength = edgeMap[x, y];

                    float gradMag = grad.magnitude;
                    Vector2 edgeNormal = gradMag > 1e-5f ? grad / gradMag : Vector2.zero;

                    // Balloon
                    float flatness = Mathf.Clamp01(1f - edgeStrength);
                    float dampening = Mathf.Clamp01(flatness * flatness);
                    float alignment = gradMag > 1e-5f ? Mathf.Max(0f, Vector2.Dot(normal, edgeNormal)) : 0f;

                    Vector2 balloonForce = (edgeStrength > EdgeThreshold)
                        ? Vector2.zero
                        : MovementPerFrame * dampening * alignment * normal;

                    // Edge attraction
                    Vector2 edgeForce = EdgeAttraction * grad;

                    // Barrier to avoid overstepping edges
                    Vector2 barrierForce = -BarrierWeight * edgeStrength * grad;

                    Vector2 force = elasticity + curvature + balloonForce + edgeForce + barrierForce;

                    // Directional lock
                    float edgeDamping = Mathf.Exp(-6.0f * edgeStrength);

                    // Velocity
                    Vector2 velocity = PositionScaling * edgeDamping * force;

                    // Remove cross-edge motion
                    if (edgeStrength > EdgeThreshold && gradMag > 1e-5f) {
                        float normalMotion = Vector2.Dot(velocity, edgeNormal);
                        velocity -= normalMotion * edgeNormal;
                    }

                    var newP = p + velocity;

                    // Clamp to image bounds
                    newP.x = Mathf.Clamp(newP.x, 0, width - 1);
                    newP.y = Mathf.Clamp(newP.y, 0, height - 1);

                    // Track movement for stabilization detection
                    totalMovement += Vector2.Distance(p, newP);

                    newPoints.Add(newP);
                }

                points = newPoints;

                // Check for stabilization - stop early if snake has stabilized
                float avgMovement = totalMovement / n;
                if (avgMovement < StabilizationThreshold) {
                    break;
                }

                // Redistribute points every 10 iterations to maintain uniform spacing
                if (iter % 10 == 0 && iter > 0) {
                    points = RedistributePoints(points, n);
                }
            }

            SEvolveSnakeMarker.End();
            return points;
        }

        private static List<Vector2> RedistributePoints(List<Vector2> points, int targetCount) {
            if (points.Count < 3) return points;

            float totalLength = 0f;
            for (int i = 0; i < points.Count; i++) {
                Vector2 p1 = points[i];
                Vector2 p2 = points[(i + 1) % points.Count];
                totalLength += Vector2.Distance(p1, p2);
            }

            float targetSpacing = totalLength / targetCount;
            List<Vector2> redistributed = new List<Vector2>(targetCount);

            redistributed.Add(points[0]);

            float accumulatedDist = 0f;
            int currentSegment = 0;

            for (int i = 1; i < targetCount; i++) {
                float targetDist = i * targetSpacing;

                // Find the segment containing the target distance
                while (currentSegment < points.Count && accumulatedDist < targetDist) {
                    Vector2 p1 = points[currentSegment];
                    Vector2 p2 = points[(currentSegment + 1) % points.Count];
                    float segmentLength = Vector2.Distance(p1, p2);

                    if (accumulatedDist + segmentLength >= targetDist) {
                        // Interpolate within segment
                        float t = (targetDist - accumulatedDist) / segmentLength;
                        Vector2 newPoint = Vector2.Lerp(p1, p2, t);
                        redistributed.Add(newPoint);
                        break;
                    }

                    accumulatedDist += segmentLength;
                    currentSegment++;
                }
            }

            return redistributed.Count > 0 ? redistributed : points;
        }

        private static Vector2[,] ComputeGradientField(float[,] edgeMap, int width, int height) {
            Vector2[,] gradient = new Vector2[width, height];

            // Compute gradient of edge map
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    float gx = (edgeMap[x + 1, y] - edgeMap[x - 1, y]) * 0.5f;
                    float gy = (edgeMap[x, y + 1] - edgeMap[x, y - 1]) * 0.5f;
                    gradient[x, y] = new Vector2(gx, gy);
                }
            }

            return gradient;
        }

        private static HashSet<int> FillContour(List<Vector2> contour, int width) {
            SFillContourMarker.Begin();
            
            if (contour.Count < 3) {
                SFillContourMarker.End();
                return new HashSet<int>();
            }

            // Find bounding box
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;

            foreach (Vector2 p in contour) {
                minX = Mathf.Min(minX, p.x);
                maxX = Mathf.Max(maxX, p.x);
                minY = Mathf.Min(minY, p.y);
                maxY = Mathf.Max(maxY, p.y);
            }

            // Estimate capacity based on bounding box area to avoid rehashing
            int estimatedCapacity = (int)((maxX - minX) * (maxY - minY));
            HashSet<int> mask = new HashSet<int>(estimatedCapacity);

            ScanlineFillMask(mask, contour, width, (int)minY, (int)maxY, (int)minX, (int)maxX);

            SFillContourMarker.End();
            return mask;
        }

        private static void ScanlineFillMask(HashSet<int> mask, List<Vector2> contour, int width, int yMin, int yMax,
            int xMin,
            int xMax) {
            // Collect all indices in a lock-free concurrent bag, then bulk-add to HashSet
            var allIndices = new ConcurrentBag<int>();
            var partitions = Partitioner.Create(yMin, yMax + 1);
            
            Parallel.ForEach(partitions, range => {
                // Use pooled intersection list to avoid allocations
                var intersections = IntersectionPool.Value;
                intersections.Clear();
                
                // Pre-allocate local capacity estimate
                int rangeSize = range.Item2 - range.Item1;
                int estimatedPixels = rangeSize * (xMax - xMin) / 4;
                var localIndices = new List<int>(estimatedPixels);

                for (int y = range.Item1; y < range.Item2; y++) {
                    FillEdgeIntersectionList(intersections, contour, y);

                    // Sort intersections and fill between pairs of intersections
                    intersections.Sort();

                    for (int i = 0; i < intersections.Count - 1; i += 2) {
                        int xStart = Mathf.Max(xMin, Mathf.CeilToInt(intersections[i]));
                        int xEnd = Mathf.Min(xMax, Mathf.FloorToInt(intersections[i + 1]));

                        for (int x = xStart; x <= xEnd; x++) {
                            localIndices.Add(y * width + x);
                        }
                    }
                    
                    intersections.Clear();
                }

                // Add all local indices to the concurrent bag
                foreach (var idx in localIndices) {
                    allIndices.Add(idx);
                }
            });
            
            // Bulk-add all indices to the HashSet
            foreach (var idx in allIndices) {
                mask.Add(idx);
            }
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
                if ((y1 < scanY && y2 < scanY)
                    || (y1 > scanY && y2 > scanY)) continue;

                // Intersection calculation
                float t = (scanY - y1) / (y2 - y1);
                float x = p1.x + t * (p2.x - p1.x);
                intersections.Add(x);
            }
        }
    }
}