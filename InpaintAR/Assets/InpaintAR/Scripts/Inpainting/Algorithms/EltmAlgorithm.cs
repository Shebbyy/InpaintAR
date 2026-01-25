using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using InpaintAR.Scripts.Util;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // Exemplar-based Local Texture Matching (ELTM) Inpainting Algorithm
    // See DOI 10.1371/journal.pone.0141199
    public class EltmAlgorithm : AbstractInpaintingAlgorithm {
        private const int PatchRadius = 4;

        // Local search window radius (paper suggests w=30-70)
        private const int SearchRadius = 30;

        // Downscale factor for faster processing
        private const int DownscaleFactor = 4;

        // Threshold for switching from geometry phase to texture phase
        private const float GeometryThreshold = 0.01f;

        private int m_width;
        private int m_height;
        private int m_pixelCount;

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            int origWidth = TextureUtility.GetImageWidth(source);
            int origHeight = TextureUtility.GetImageHeight(source);

            // Downscale for faster processing
            m_width = origWidth / DownscaleFactor;
            m_height = origHeight / DownscaleFactor;
            m_pixelCount = m_width * m_height;

            // Downscale source pixels
            var downscaledPixels = DownscalePixels(MPixelBuffer, origWidth, origHeight, m_width, m_height);

            // Downscale mask indices
            var downscaledMask = new HashSet<int>();
            foreach (int idx in maskPixelIndices) {
                int origX = idx % origWidth;
                int origY = idx / origWidth;
                int newX = origX / DownscaleFactor;
                int newY = origY / DownscaleFactor;
                if (newX < m_width && newY < m_height) {
                    downscaledMask.Add(newY * m_width + newX);
                }
            }

            var pixels = new NativeArray<Color32>(downscaledPixels, Allocator.TempJob);
            var confidence = new NativeArray<float>(m_pixelCount, Allocator.TempJob);
            var maskSet = new NativeParallelHashSet<int>(downscaledMask.Count, Allocator.TempJob);

            // Copy mask indices to hash set
            foreach (int idx in downscaledMask) {
                maskSet.Add(idx);
            }

            // Initialize confidence: 1 for known, 0 for unknown
            var initJob = new InitializeConfidenceJob {
                Confidence = confidence,
                MaskSet = maskSet
            };
            initJob.Schedule(m_pixelCount, 64).Complete();

            // Run ELTM inpainting
            InpaintEltmBurst(ref pixels, ref confidence, ref maskSet);

            var inpaintedSmall = new Color32[m_pixelCount];
            pixels.CopyTo(inpaintedSmall);

            pixels.Dispose();
            confidence.Dispose();
            maskSet.Dispose();

            // Upscale result back to original size
            var upscaledPixels = UpscalePixels(inpaintedSmall, m_width, m_height, origWidth, origHeight);

            // Blend: use inpainted pixels only where mask was
            foreach (var i in maskPixelIndices) {
                MPixelBuffer[i] = upscaledPixels[i];
            }

            Texture2D resultImage = new Texture2D(origWidth, origHeight, TextureFormat.RGBA32, false);
            resultImage.SetPixels32(MPixelBuffer);
            resultImage.Apply();

            return resultImage;
        }

        private static Color32[] DownscalePixels(Color32[] source, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
            var result = new Color32[dstWidth * dstHeight];
            float scaleX = (float)srcWidth / dstWidth;
            float scaleY = (float)srcHeight / dstHeight;

            for (int y = 0; y < dstHeight; y++) {
                for (int x = 0; x < dstWidth; x++) {
                    int srcX = (int)(x * scaleX);
                    int srcY = (int)(y * scaleY);
                    result[y * dstWidth + x] = source[srcY * srcWidth + srcX];
                }
            }
            return result;
        }

        private static Color32[] UpscalePixels(Color32[] source, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
            var result = new Color32[dstWidth * dstHeight];
            float scaleX = (float)(srcWidth - 1) / (dstWidth - 1);
            float scaleY = (float)(srcHeight - 1) / (dstHeight - 1);

            for (int y = 0; y < dstHeight; y++) {
                for (int x = 0; x < dstWidth; x++) {
                    float srcXf = x * scaleX;
                    float srcYf = y * scaleY;
                    int x0 = (int)srcXf;
                    int y0 = (int)srcYf;
                    int x1 = math.min(x0 + 1, srcWidth - 1);
                    int y1 = math.min(y0 + 1, srcHeight - 1);
                    float fx = srcXf - x0;
                    float fy = srcYf - y0;

                    Color32 c00 = source[y0 * srcWidth + x0];
                    Color32 c10 = source[y0 * srcWidth + x1];
                    Color32 c01 = source[y1 * srcWidth + x0];
                    Color32 c11 = source[y1 * srcWidth + x1];

                    result[y * dstWidth + x] = new Color32(
                        (byte)math.lerp(math.lerp(c00.r, c10.r, fx), math.lerp(c01.r, c11.r, fx), fy),
                        (byte)math.lerp(math.lerp(c00.g, c10.g, fx), math.lerp(c01.g, c11.g, fx), fy),
                        (byte)math.lerp(math.lerp(c00.b, c10.b, fx), math.lerp(c01.b, c11.b, fx), fy),
                        255
                    );
                }
            }
            return result;
        }

        [BurstCompile]
        private struct InitializeConfidenceJob : IJobParallelFor {
            [WriteOnly] public NativeArray<float> Confidence;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;

            public void Execute(int index) {
                Confidence[index] = MaskSet.Contains(index) ? 0f : 1f;
            }
        }

        private void InpaintEltmBurst(
            ref NativeArray<Color32> pixels,
            ref NativeArray<float> confidence,
            ref NativeParallelHashSet<int> maskSet) {

            var contourPixels = new NativeList<int>(Allocator.Temp);
            bool geometryPhase = true;

            while (!maskSet.IsEmpty) {
                // Step 1: Find contour pixels (parallel)
                contourPixels.Clear();
                GetContourPixelsParallel(ref maskSet, ref contourPixels, m_width, m_height);

                if (contourPixels.Length == 0) break;

                // Step 2: Find target patch with highest priority (parallel)
                int targetIndex;
                if (geometryPhase) {
                    targetIndex = GetTargetGeometryParallel(
                        ref contourPixels, ref maskSet, ref pixels,
                        m_width, m_height, out var maxDataTerm);

                    if (maxDataTerm < GeometryThreshold) {
                        geometryPhase = false;
                    }
                } else {
                    targetIndex = GetTargetTextureParallel(
                        ref contourPixels, ref confidence, m_width, m_height);
                }

                if (targetIndex < 0) break;

                int targetX = targetIndex % m_width;
                int targetY = targetIndex / m_width;

                // Step 3: Find best matching patch (parallel)
                int bestPatch = FindBestMatchParallel(
                    targetX, targetY, ref maskSet, ref pixels, m_width, m_height);

                // Step 4: Fill target patch
                if (bestPatch >= 0) {
                    FillPatchFromSource(targetX, targetY, bestPatch, ref maskSet, ref pixels, m_width, m_height);
                } else {
                    FillPatchFromNearest(targetX, targetY, ref maskSet, ref pixels, m_width, m_height);
                }

                // Step 5: Update confidence and remove filled pixels
                UpdateConfidence(targetX, targetY, ref confidence, ref maskSet, m_width, m_height);
            }

            contourPixels.Dispose();
        }

        [BurstCompile]
        private struct ContourJob : IJobParallelFor {
            [ReadOnly] public NativeArray<int> MaskArray;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;
            [ReadOnly] public int Width;
            [ReadOnly] public int PixelCount;
            [WriteOnly] public NativeArray<bool> IsContour;

            public void Execute(int i) {
                int index = MaskArray[i];
                int x = index % Width;

                bool isContour = x > 0 && !MaskSet.Contains(index - 1);
                if (!isContour && x < Width - 1 && !MaskSet.Contains(index + 1)) isContour = true;
                if (!isContour && index >= Width && !MaskSet.Contains(index - Width)) isContour = true;
                if (!isContour && index < PixelCount - Width && !MaskSet.Contains(index + Width)) isContour = true;

                IsContour[i] = isContour;
            }
        }

        private static void GetContourPixelsParallel(
            ref NativeParallelHashSet<int> maskSet,
            ref NativeList<int> contourPixels,
            int width, int height) {

            int pixelCount = width * height;
            var maskArray = maskSet.ToNativeArray(Allocator.TempJob);
            int maskCount = maskArray.Length;

            if (maskCount == 0) {
                maskArray.Dispose();
                return;
            }

            var isContour = new NativeArray<bool>(maskCount, Allocator.TempJob);

            var job = new ContourJob {
                MaskArray = maskArray,
                MaskSet = maskSet,
                Width = width,
                PixelCount = pixelCount,
                IsContour = isContour
            };

            job.Schedule(maskCount, 64).Complete();

            for (int i = 0; i < maskCount; i++) {
                if (isContour[i]) {
                    contourPixels.Add(maskArray[i]);
                }
            }

            maskArray.Dispose();
            isContour.Dispose();
        }

        [BurstCompile]
        private struct GeometryPriorityJob : IJobParallelFor {
            [ReadOnly] public NativeArray<int> ContourPixels;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;
            [ReadOnly] public NativeArray<Color32> Pixels;
            [ReadOnly] public int Width;
            [ReadOnly] public int Height;
            [WriteOnly] public NativeArray<float> Priorities;

            public void Execute(int i) {
                int pixel = ContourPixels[i];
                int x = pixel % Width;
                int y = pixel / Width;

                float2 normal = EstimateNormalStatic(x, y, MaskSet, Width, Height);
                Priorities[i] = CalculateDataStatic(x, y, normal, MaskSet, Pixels, Width, Height);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static float2 EstimateNormalStatic(int px, int py, NativeParallelHashSet<int> maskSet, int width, int height) {
                float GetMask(int x, int y) {
                    if (x < 0 || x >= width || y < 0 || y >= height) return 0f;
                    return maskSet.Contains(y * width + x) ? 1f : 0f;
                }

                float dx = (GetMask(px + 1, py) - GetMask(px - 1, py)) / 2f;
                float dy = (GetMask(px, py + 1) - GetMask(px, py - 1)) / 2f;

                float2 normal = new float2(dx, dy);
                float sqrMag = normal.x * normal.x + normal.y * normal.y;
                return sqrMag > 1e-6f ? math.normalize(normal) : new float2(0f, 1f);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static float CalculateDataStatic(int px, int py, float2 normal, NativeParallelHashSet<int> maskSet, NativeArray<Color32> pixels, int width, int height) {
                int centerIdx = py * width + px;
                float centerGray = (pixels[centerIdx].r * 0.299f + pixels[centerIdx].g * 0.587f + pixels[centerIdx].b * 0.114f) / 255f;

                float GetVal(int x, int y) {
                    if (x < 0 || x >= width || y < 0 || y >= height) return centerGray;
                    int idx = y * width + x;
                    if (maskSet.Contains(idx)) return centerGray;
                    return (pixels[idx].r * 0.299f + pixels[idx].g * 0.587f + pixels[idx].b * 0.114f) / 255f;
                }

                float gx = -GetVal(px-1, py-1) + GetVal(px+1, py-1) - 2*GetVal(px-1, py) + 2*GetVal(px+1, py) - GetVal(px-1, py+1) + GetVal(px+1, py+1);
                float gy = GetVal(px-1, py-1) + 2*GetVal(px, py-1) + GetVal(px+1, py-1) - GetVal(px-1, py+1) - 2*GetVal(px, py+1) - GetVal(px+1, py+1);

                float2 isophote = new float2(-gy, gx);
                return math.abs(isophote.x * normal.x + isophote.y * normal.y) / 255f + 0.001f;
            }
        }

        private static int GetTargetGeometryParallel(
            ref NativeList<int> contourPixels,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height,
            out float maxDataTerm) {

            int count = contourPixels.Length;
            maxDataTerm = 0f;
            if (count == 0) return -1;

            var priorities = new NativeArray<float>(count, Allocator.TempJob);

            var job = new GeometryPriorityJob {
                ContourPixels = contourPixels.AsArray(),
                MaskSet = maskSet,
                Pixels = pixels,
                Width = width,
                Height = height,
                Priorities = priorities
            };

            job.Schedule(count, 64).Complete();

            float maxPriority = -1f;
            int bestIndex = -1;
            for (int i = 0; i < count; i++) {
                if (priorities[i] > maxDataTerm) maxDataTerm = priorities[i];
                if (priorities[i] > maxPriority) {
                    maxPriority = priorities[i];
                    bestIndex = contourPixels[i];
                }
            }

            priorities.Dispose();
            return bestIndex;
        }

        [BurstCompile]
        private struct TexturePriorityJob : IJobParallelFor {
            [ReadOnly] public NativeArray<int> ContourPixels;
            [ReadOnly] public NativeArray<float> Confidence;
            [ReadOnly] public int Width;
            [ReadOnly] public int Height;
            [WriteOnly] public NativeArray<float> Priorities;

            public void Execute(int i) {
                int pixel = ContourPixels[i];
                int px = pixel % Width;
                int py = pixel / Width;

                float sum = 0f;
                int cnt = 0;
                for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                    for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                        int nx = px + dx;
                        int ny = py + dy;
                        if (nx >= 0 && nx < Width && ny >= 0 && ny < Height) {
                            sum += Confidence[ny * Width + nx];
                            cnt++;
                        }
                    }
                }
                Priorities[i] = cnt > 0 ? sum / cnt : 0f;
            }
        }

        private static int GetTargetTextureParallel(
            ref NativeList<int> contourPixels,
            ref NativeArray<float> confidence,
            int width, int height) {

            int count = contourPixels.Length;
            if (count == 0) return -1;

            var priorities = new NativeArray<float>(count, Allocator.TempJob);

            var job = new TexturePriorityJob {
                ContourPixels = contourPixels.AsArray(),
                Confidence = confidence,
                Width = width,
                Height = height,
                Priorities = priorities
            };

            job.Schedule(count, 64).Complete();

            float maxPriority = -1f;
            int bestIndex = -1;
            for (int i = 0; i < count; i++) {
                if (priorities[i] > maxPriority) {
                    maxPriority = priorities[i];
                    bestIndex = contourPixels[i];
                }
            }

            priorities.Dispose();
            return bestIndex;
        }

        [BurstCompile]
        private struct LocalSearchJob : IJobParallelFor {
            [ReadOnly] public int TargetX;
            [ReadOnly] public int TargetY;
            [ReadOnly] public NativeArray<int> SearchPositions;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;
            [ReadOnly] public NativeArray<Color32> Pixels;
            [ReadOnly] public int Width;
            [ReadOnly] public int Height;
            [WriteOnly] public NativeArray<float> Distances;

            public void Execute(int i) {
                int pos = SearchPositions[i];
                int sx = pos % Width;
                int sy = pos / Width;

                if (!IsPatchFullyKnownStatic(sx, sy, MaskSet, Width, Height)) {
                    Distances[i] = float.MaxValue;
                    return;
                }

                Distances[i] = CalculateSsdStatic(TargetX, TargetY, sx, sy, MaskSet, Pixels, Width, Height);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static bool IsPatchFullyKnownStatic(int cx, int cy, NativeParallelHashSet<int> maskSet, int width, int height) {
                for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                    for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                        int nx = cx + dx;
                        int ny = cy + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) return false;
                        if (maskSet.Contains(ny * width + nx)) return false;
                    }
                }
                return true;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static float CalculateSsdStatic(int targetX, int targetY, int sourceX, int sourceY,
                NativeParallelHashSet<int> maskSet, NativeArray<Color32> pixels, int width, int height) {

                float sum = 0f;
                int cnt = 0;

                for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                    for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                        int tx = targetX + dx;
                        int ty = targetY + dy;
                        if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

                        int targetIdx = ty * width + tx;
                        if (maskSet.Contains(targetIdx)) continue;

                        int sx = sourceX + dx;
                        int sy = sourceY + dy;
                        int sourceIdx = sy * width + sx;

                        Color32 tc = pixels[targetIdx];
                        Color32 sc = pixels[sourceIdx];

                        float dr = tc.r - sc.r;
                        float dg = tc.g - sc.g;
                        float db = tc.b - sc.b;

                        sum += dr * dr + dg * dg + db * db;
                        cnt++;
                    }
                }

                return cnt > 0 ? sum / cnt : float.MaxValue;
            }
        }

        private static int FindBestMatchParallel(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            int minX = math.max(PatchRadius, targetX - SearchRadius);
            int maxX = math.min(width - PatchRadius - 1, targetX + SearchRadius);
            int minY = math.max(PatchRadius, targetY - SearchRadius);
            int maxY = math.min(height - PatchRadius - 1, targetY + SearchRadius);

            // Collect search positions using boolean array
            int pixelCount = width * height;
            var visited = new NativeArray<bool>(pixelCount, Allocator.Temp);

            for (int sy = minY; sy <= maxY; sy++) {
                for (int sx = minX; sx <= maxX; sx++) {
                    if (math.abs(sx - targetX) <= PatchRadius && math.abs(sy - targetY) <= PatchRadius)
                        continue;
                    visited[sy * width + sx] = true;
                }
            }

            int totalPositions = 0;
            for (int i = 0; i < pixelCount; i++) {
                if (visited[i]) totalPositions++;
            }

            if (totalPositions == 0) {
                visited.Dispose();
                return -1;
            }

            var positionsArray = new NativeArray<int>(totalPositions, Allocator.TempJob);
            int writeIdx = 0;
            for (int i = 0; i < pixelCount; i++) {
                if (visited[i]) positionsArray[writeIdx++] = i;
            }
            visited.Dispose();

            var distances = new NativeArray<float>(totalPositions, Allocator.TempJob);

            var job = new LocalSearchJob {
                TargetX = targetX,
                TargetY = targetY,
                SearchPositions = positionsArray,
                MaskSet = maskSet,
                Pixels = pixels,
                Width = width,
                Height = height,
                Distances = distances
            };

            job.Schedule(totalPositions, 64).Complete();

            float bestDist = float.MaxValue;
            int bestPatch = -1;
            for (int i = 0; i < totalPositions; i++) {
                if (distances[i] < bestDist) {
                    bestDist = distances[i];
                    bestPatch = positionsArray[i];
                }
            }

            positionsArray.Dispose();
            distances.Dispose();

            return bestPatch;
        }

        [BurstCompile]
        private static void FillPatchFromSource(
            int targetX, int targetY, int sourcePatchIdx,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            int sourceX = sourcePatchIdx % width;
            int sourceY = sourcePatchIdx / width;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

                    int targetIdx = ty * width + tx;
                    if (!maskSet.Contains(targetIdx)) continue;

                    int sx = sourceX + dx;
                    int sy = sourceY + dy;

                    pixels[targetIdx] = pixels[sy * width + sx];
                }
            }
        }

        [BurstCompile]
        private static void FillPatchFromNearest(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            int nearestIdx = -1;
            float nearestDist = float.MaxValue;

            for (int dy = -PatchRadius * 2; dy <= PatchRadius * 2; dy++) {
                for (int dx = -PatchRadius * 2; dx <= PatchRadius * 2; dx++) {
                    int nx = targetX + dx;
                    int ny = targetY + dy;

                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                    int idx = ny * width + nx;
                    if (maskSet.Contains(idx)) continue;

                    float dist = dx * dx + dy * dy;
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        nearestIdx = idx;
                    }
                }
            }

            if (nearestIdx < 0) return;

            Color32 fillColor = pixels[nearestIdx];

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

                    int targetIdx = ty * width + tx;
                    if (!maskSet.Contains(targetIdx)) continue;

                    pixels[targetIdx] = fillColor;
                }
            }
        }

        [BurstCompile]
        private static void UpdateConfidence(
            int targetX, int targetY,
            ref NativeArray<float> confidence,
            ref NativeParallelHashSet<int> maskSet,
            int width, int height) {

            float sum = 0f;
            int cnt = 0;
            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int nx = targetX + dx;
                    int ny = targetY + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += confidence[ny * width + nx];
                        cnt++;
                    }
                }
            }
            float patchConfidence = cnt > 0 ? sum / cnt : 0f;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

                    int targetIdx = ty * width + tx;
                    if (!maskSet.Contains(targetIdx)) continue;

                    confidence[targetIdx] = patchConfidence;
                    maskSet.Remove(targetIdx);
                }
            }
        }
    }
}
