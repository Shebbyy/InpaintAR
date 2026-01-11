using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Unity.Profiling;
using UnityEngine;

namespace InpaintAR.Scripts.Util {
    public static class TextureUtility {
        // Thread-local pool for intersection lists to avoid allocations
        private static readonly ThreadLocal<List<float>> IntersectionPool = new(() => new List<float>(100));
        private static readonly ThreadLocal<List<int>> LocalIndicesPool = new(() => new List<int>(100));
        
        private static readonly ProfilerMarker SFillContourMarker = new("TextureUtility.FillContour");
        
        
        // Size should always stay the same during program, only mask changes; -> Initialize empty once and then reuse to avoid reinitialization each frame
        private static int _mImageWidth = -1;
        private static int _mImageHeight = -1;

        private static Color[] _mEmptyImage;

        public static Color[] GetEmptyImagePixels(Texture fillImageTexture) {
            if (_mEmptyImage is null) {
                InitializeEmptyPixels(fillImageTexture);
            }

            return _mEmptyImage;
        }
        
        public static void FillContour(HashSet<int> mask, List<Vector2> contour, int width) {
            SFillContourMarker.Begin();
            mask.Clear();
            
            if (contour.Count < 3) {
                SFillContourMarker.End();
                return;
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

            ScanlineFillMask(mask, contour, width, (int)minY, (int)maxY, (int)minX, (int)maxX);

            SFillContourMarker.End();
        }

        public static int GetImageWidth(Texture fillImageTexture) {
            if (_mEmptyImage is null) {
                InitializeEmptyPixels(fillImageTexture);
            }

            return _mImageWidth;
        }
        
        public static int GetImageHeight(Texture fillImageTexture) {
            if (_mEmptyImage is null) {
                InitializeEmptyPixels(fillImageTexture);
            }

            return _mImageHeight;
        }
        
        private static void InitializeEmptyPixels(Texture fillImageTexture) {
            _mImageWidth  = fillImageTexture.width;
            _mImageHeight = fillImageTexture.height;
            _mEmptyImage  = new Color[_mImageWidth * _mImageHeight];

            for (int i = 0; i < _mEmptyImage.Length; i++) {
                _mEmptyImage[i] = Color.clear;
            }
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
                
                var localIndices = LocalIndicesPool.Value;
                localIndices.Clear();

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
        
        public static List<Vector2> RedistributePoints(List<Vector2> points, int targetCount) {
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

        public static Texture2D CopyTexture(Texture source) {
            var newTexture = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);

            RenderTexture currentRT = RenderTexture.active;

            // Source is usually RenderTexture when delivered from Quest
            RenderTexture texture = source as RenderTexture;
            RenderTexture.active = texture;

            newTexture.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            newTexture.Apply();

            RenderTexture.active = currentRT;

            return newTexture;
        }
    }
}