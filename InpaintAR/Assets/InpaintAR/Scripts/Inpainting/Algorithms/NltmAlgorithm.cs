using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using InpaintAR.Scripts.Util;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // Nonlocal Texture Matching (NLTM) Inpainting Algorithm
    // See DOI 10.1109/TIP.2018.2880681
    // Extended with PatchMatch-inspired temporal coherence for real-time performance
    public class NltmAlgorithm : AbstractInpaintingAlgorithm {
        private const int PatchRadius = 4;
        private const int PatchSize = 2 * PatchRadius + 1;
        private const int PatchArea = PatchSize * PatchSize;

        // Downscale factor for faster processing (4 = quarter resolution)
        private const int DownscaleFactor = 6;

        // Number of candidate patches K (5-10 recommended according to paper)
        private const int K = 5;

        // Alpha for trimmed mean filter (10-20% recommended by paper to be sliced from each end)
        private const float Alpha = 0.15f;

        // Gaussian sigma for texture matching weight
        private const float GaussianSigma = PatchRadius / 2.0f;

        // Local search radius around cached candidates (PatchMatch-inspired)
        private const int LocalSearchRadius = 10;

        private int m_width;
        private int m_height;
        private int m_pixelCount;

        // Temporal cache for candidate positions
        private readonly List<int> m_cachedCandidates = new();

        private int m_prevMaskLength = -1;

        // Persistent native arrays (reused across frames to avoid allocation)
        private NativeArray<float> m_gaussianWeights;
        private bool m_gaussianWeightsInitialized;

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            int origWidth = TextureUtility.GetImageWidth(source);
            int origHeight = TextureUtility.GetImageHeight(source);

            // Downscale for faster processing
            m_width = origWidth / DownscaleFactor;
            m_height = origHeight / DownscaleFactor;
            m_pixelCount = m_width * m_height;

            // Downscale source pixels
            var downscaledPixels = DownscalePixels(PixelBuffer, origWidth, origHeight, m_width, m_height);

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

            // If mask changed by at least 10%, invalidate cache
            int curCount = downscaledMask.Count;
            if (m_prevMaskLength > 0) {
                float changeRatio = math.abs(curCount - m_prevMaskLength) / (float)m_prevMaskLength;
                if (changeRatio >= 0.1f) {
                    InvalidateCache();
                    m_prevMaskLength = curCount;
                }
            } else {
                m_prevMaskLength = curCount;
            }

            // Precompute Gaussian weights (only once)
            if (!m_gaussianWeightsInitialized) {
                PrecomputeGaussianWeights();
            }

            // Convert to native containers for Burst
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

            // Convert cached candidates to native
            var cachedCandidates = new NativeList<int>(m_cachedCandidates.Count, Allocator.TempJob);
            foreach (int c in m_cachedCandidates) {
                cachedCandidates.Add(c);
            }

            // Run NLTM inpainting
            var newCandidates = new NativeList<int>(Allocator.TempJob);
            InpaintNltmBurst(ref pixels, ref confidence, ref maskSet, ref cachedCandidates, ref newCandidates);

            // Update cache for next frame
            m_cachedCandidates.Clear();
            foreach (var t in newCandidates) {
                m_cachedCandidates.Add(t);
            }

            // Copy results back to temp buffer
            var inpaintedSmall = new Color32[m_pixelCount];
            pixels.CopyTo(inpaintedSmall);

            // Dispose native containers
            pixels.Dispose();
            confidence.Dispose();
            maskSet.Dispose();
            cachedCandidates.Dispose();
            newCandidates.Dispose();

            // Upscale result back to original size
            var upscaledPixels = UpscalePixels(inpaintedSmall, m_width, m_height, origWidth, origHeight);

            // Blend: use inpainted pixels only where mask was, keep original elsewhere
            foreach (var i in maskPixelIndices) {
                PixelBuffer[i] = upscaledPixels[i];
            }
            
            // Create result texture at original size
            Texture2D resultImage = new Texture2D(origWidth, origHeight, TextureFormat.RGBA32, false);
            resultImage.SetPixels32(PixelBuffer);
            resultImage.Apply();

            return resultImage;
        }

        private static Color32[] DownscalePixels(Color32[] source, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
            var result = new Color32[dstWidth * dstHeight];
            float scaleX = (float)srcWidth / dstWidth;
            float scaleY = (float)srcHeight / dstHeight;

            for (int y = 0; y < dstHeight; y++) {
                for (int x = 0; x < dstWidth; x++) {
                    // Average pixels in the source block
                    int srcX = (int)(x * scaleX);
                    int srcY = (int)(y * scaleY);
                    int srcIdx = srcY * srcWidth + srcX;
                    result[y * dstWidth + x] = source[srcIdx];
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

                    // Bilinear interpolation
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

        private void InvalidateCache() {
            Debug.Log("NLTM Cache Cleared");
            m_cachedCandidates.Clear();
            m_gaussianWeightsInitialized = false;
        }

        private void PrecomputeGaussianWeights() {
            m_gaussianWeights = new NativeArray<float>(PatchArea, Allocator.Persistent);
            float sigma2 = 2f * GaussianSigma * GaussianSigma;
            int idx = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    float distSq = dx * dx + dy * dy;
                    m_gaussianWeights[idx++] = math.exp(-distSq / sigma2);
                }
            }
            m_gaussianWeightsInitialized = true;
        }

        ~NltmAlgorithm() {
            if (m_gaussianWeightsInitialized && m_gaussianWeights.IsCreated) {
                m_gaussianWeights.Dispose();
            }
        }

        private void InpaintNltmBurst(
            ref NativeArray<Color32> pixels,
            ref NativeArray<float> confidence,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeList<int> cachedCandidates,
            ref NativeList<int> outNewCandidates) {

            var contourPixels = new NativeList<int>(Allocator.Temp);
            var candidates = new NativeList<CandidateInfo>(Allocator.Temp);
            var candidatePatches = new NativeList<int>(K, Allocator.Temp);

            while (!maskSet.IsEmpty) {
                // Step 1: Find contour pixels (parallel)
                contourPixels.Clear();
                GetContourPixelsParallel(ref maskSet, ref contourPixels, m_width, m_height);

                if (contourPixels.Length == 0) break;

                // Step 2: Find target patch with highest priority (parallel)
                int targetIndex = GetTargetPatchParallel(
                    ref contourPixels, ref confidence, ref maskSet, ref pixels,
                    m_width, m_height);

                if (targetIndex < 0) break;

                int targetX = targetIndex % m_width;
                int targetY = targetIndex / m_width;

                // Step 3: Find K best candidate patches
                candidates.Clear();
                candidatePatches.Clear();

                bool useCached = cachedCandidates.Length > 0;
                if (useCached) {
                    SearchCachedParallel(
                        targetX, targetY, ref maskSet, ref pixels, ref cachedCandidates,
                        ref candidates, m_gaussianWeights, m_width, m_height);
                } else {
                    SearchFullImageBurst(
                        targetX, targetY, ref maskSet, ref pixels, ref candidates,
                        m_gaussianWeights, m_width, m_height);
                }

                // Sort candidates and take K best
                SortCandidates(ref candidates);
                int count = math.min(K, candidates.Length);
                for (int i = 0; i < count; i++) {
                    candidatePatches.Add(candidates[i].Index);
                }

                // Step 4: Fill target patch
                if (candidatePatches.Length == 0) {
                    FillPatchFromNearestBurst(targetX, targetY, ref maskSet, ref pixels, m_width, m_height);
                } else {
                    FillPatchWithTrimmedMeanBurst(
                        targetX, targetY, ref candidatePatches, ref maskSet, ref pixels, m_width, m_height);
                    
                    // Store candidates for next frame's cache
                    foreach (var t1 in candidatePatches) {
                        bool found = false;
                        foreach (var t in outNewCandidates) {
                            if (t != t1) continue;
                            
                            found = true;
                            break;
                        }
                        if (!found) {
                            outNewCandidates.Add(t1);
                        }
                    }
                }

                // Step 5: Update confidence and remove filled pixels from mask
                UpdateConfidenceBurst(targetX, targetY, ref confidence, ref maskSet, m_width, m_height);
            }

            contourPixels.Dispose();
            candidates.Dispose();
            candidatePatches.Dispose();
        }

        [BurstCompile]
        private struct CandidateInfo : IComparable<CandidateInfo> {
            public int Index;
            public float Distance;

            public int CompareTo(CandidateInfo other) {
                return Distance.CompareTo(other.Distance);
            }
        }

        [BurstCompile]
        private struct InitializeConfidenceJob : IJobParallelFor {
            [WriteOnly] public NativeArray<float> Confidence;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;

            public void Execute(int index) {
                // Check if any mask index matches
                Confidence[index] = MaskSet.Contains(index) ? 0f : 1f;
            }
        }

        // Parallel job for finding contour pixels
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

                // Left
                bool isContour = x > 0 && !MaskSet.Contains(index - 1);

                // Right
                if (!isContour && x < Width - 1 && !MaskSet.Contains(index + 1)) isContour = true;
                // Up
                if (!isContour && index >= Width && !MaskSet.Contains(index - Width)) isContour = true;
                // Down
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

            // Collect contour pixels
            for (int i = 0; i < maskCount; i++) {
                if (isContour[i]) {
                    contourPixels.Add(maskArray[i]);
                }
            }

            maskArray.Dispose();
            isContour.Dispose();
        }

        // Parallel job for calculating priorities
        [BurstCompile]
        private struct PriorityJob : IJobParallelFor {
            [ReadOnly] public NativeArray<int> ContourPixels;
            [ReadOnly] public NativeArray<float> Confidence;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;
            [ReadOnly] public NativeArray<Color32> Pixels;
            [ReadOnly] public int Width;
            [ReadOnly] public int Height;

            [WriteOnly] public NativeArray<float> Priorities;

            public void Execute(int i) {
                int pixel = ContourPixels[i];
                int x = pixel % Width;
                int y = pixel / Width;

                float conf = CalculateConfidenceStatic(x, y, Confidence, Width, Height);
                float2 normal = EstimateNormalStatic(x, y, MaskSet, Width, Height);
                float data = CalculateDataStatic(x, y, normal, MaskSet, Pixels, Width, Height);

                Priorities[i] = conf * data;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static float CalculateConfidenceStatic(int px, int py, NativeArray<float> confidence, int width, int height) {
                float sum = 0f;
                int count = 0;

                for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                    for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                        int nx = px + dx;
                        int ny = py + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            sum += confidence[ny * width + nx];
                            count++;
                        }
                    }
                }
                return count > 0 ? sum / count : 0f;
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
                float dot = math.abs(isophote.x * normal.x + isophote.y * normal.y);
                return dot / 255f + 0.001f;
            }
        }

        private static int GetTargetPatchParallel(
            ref NativeList<int> contourPixels,
            ref NativeArray<float> confidence,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            int count = contourPixels.Length;
            if (count == 0) return -1;

            var priorities = new NativeArray<float>(count, Allocator.TempJob);

            var job = new PriorityJob {
                ContourPixels = contourPixels.AsArray(),
                Confidence = confidence,
                MaskSet = maskSet,
                Pixels = pixels,
                Width = width,
                Height = height,
                Priorities = priorities
            };

            job.Schedule(count, 64).Complete();

            // Find max priority
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CalculateConfidenceBurst(int px, int py, ref NativeArray<float> confidence, int width, int height) {
            float sumConfidence = 0f;
            int validPixels = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int nx = px + dx;
                    int ny = py + dy;

                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                    int idx = ny * width + nx;
                    sumConfidence += confidence[idx];
                    validPixels++;
                }
            }

            return validPixels > 0 ? sumConfidence / validPixels : 0f;
        }
        
        private static void SearchCachedParallel(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            ref NativeList<int> cachedCandidates,
            ref NativeList<CandidateInfo> candidates,
            NativeArray<float> gaussianWeights,
            int width, int height) {

            int numCached = cachedCandidates.Length;
            if (numCached == 0) return;

            int pixelCount = width * height;
            var visited = new NativeArray<bool>(pixelCount, Allocator.Temp);

            // First pass: mark all positions to visit
            for (int c = 0; c < numCached; c++) {
                int cachedPos = cachedCandidates[c];
                int cachedX = cachedPos % width;
                int cachedY = cachedPos / width;

                int minX = math.max(PatchRadius, cachedX - LocalSearchRadius);
                int maxX = math.min(width - PatchRadius - 1, cachedX + LocalSearchRadius);
                int minY = math.max(PatchRadius, cachedY - LocalSearchRadius);
                int maxY = math.min(height - PatchRadius - 1, cachedY + LocalSearchRadius);

                for (int sy = minY; sy <= maxY; sy++) {
                    for (int sx = minX; sx <= maxX; sx++) {
                        // Skip if overlapping target
                        if (math.abs(sx - targetX) <= PatchRadius && math.abs(sy - targetY) <= PatchRadius)
                            continue;
                        visited[sy * width + sx] = true;
                    }
                }
            }

            // Count marked positions
            int totalPositions = 0;
            for (int i = 0; i < pixelCount; i++) {
                if (visited[i]) totalPositions++;
            }

            if (totalPositions == 0) {
                visited.Dispose();
                return;
            }

            // Second pass: collect positions into array
            var positionsArray = new NativeArray<int>(totalPositions, Allocator.TempJob);
            int writeIdx = 0;
            for (int i = 0; i < pixelCount; i++) {
                if (visited[i]) {
                    positionsArray[writeIdx++] = i;
                }
            }
            visited.Dispose();

            // Run parallel job over all positions (same as full image search)
            var results = new NativeArray<CandidateInfo>(totalPositions, Allocator.TempJob);

            var job = new SearchPositionsJob {
                TargetX = targetX,
                TargetY = targetY,
                Width = width,
                Height = height,
                SearchPositions = positionsArray,
                MaskSet = maskSet,
                Pixels = pixels,
                GaussianWeights = gaussianWeights,
                Results = results
            };

            job.Schedule(totalPositions, 64).Complete();

            // Collect valid results
            for (int i = 0; i < totalPositions; i++) {
                if (results[i].Index >= 0) {
                    candidates.Add(results[i]);
                }
            }

            positionsArray.Dispose();
            results.Dispose();
        }

        // Parallel job for searching specific positions
        [BurstCompile]
        private struct SearchPositionsJob : IJobParallelFor {
            [ReadOnly] public int TargetX;
            [ReadOnly] public int TargetY;
            [ReadOnly] public int Width;
            [ReadOnly] public int Height;
            [ReadOnly] public NativeArray<int> SearchPositions;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;
            [ReadOnly] public NativeArray<Color32> Pixels;
            [ReadOnly] public NativeArray<float> GaussianWeights;

            [WriteOnly] public NativeArray<CandidateInfo> Results;

            public void Execute(int index) {
                int pos = SearchPositions[index];
                int sx = pos % Width;
                int sy = pos / Width;

                if (!IsPatchFullyKnownBurstStatic(sx, sy, MaskSet, Width, Height)) {
                    Results[index] = new CandidateInfo { Index = -1, Distance = float.MaxValue };
                    return;
                }

                float distance = CalculateTextureDistanceBurstStatic(
                    TargetX, TargetY, sx, sy, MaskSet, Pixels, GaussianWeights, Width, Height);

                if (distance >= 0) {
                    Results[index] = new CandidateInfo { Index = pos, Distance = distance };
                } else {
                    Results[index] = new CandidateInfo { Index = -1, Distance = float.MaxValue };
                }
            }
        }

        // Parallel job for searching the full image
        [BurstCompile]
        private struct SearchFullImageJob : IJobParallelFor {
            [ReadOnly] public int TargetX;
            [ReadOnly] public int TargetY;
            [ReadOnly] public int Width;
            [ReadOnly] public int Height;
            [ReadOnly] public int SearchWidth;
            [ReadOnly] public int SearchStartX;
            [ReadOnly] public int SearchStartY;
            [ReadOnly] public NativeParallelHashSet<int> MaskSet;
            [ReadOnly] public NativeArray<Color32> Pixels;
            [ReadOnly] public NativeArray<float> GaussianWeights;

            [WriteOnly] public NativeArray<CandidateInfo> Results;

            public void Execute(int index) {
                int sx = SearchStartX + (index % SearchWidth);
                int sy = SearchStartY + (index / SearchWidth);

                // Skip if overlapping with target patch
                if (math.abs(sx - TargetX) <= PatchRadius && math.abs(sy - TargetY) <= PatchRadius) {
                    Results[index] = new CandidateInfo { Index = -1, Distance = float.MaxValue };
                    return;
                }

                // Check if patch is fully known
                if (!IsPatchFullyKnownBurstStatic(sx, sy, MaskSet, Width, Height)) {
                    Results[index] = new CandidateInfo { Index = -1, Distance = float.MaxValue };
                    return;
                }

                float distance = CalculateTextureDistanceBurstStatic(
                    TargetX, TargetY, sx, sy, MaskSet, Pixels, GaussianWeights, Width, Height);

                if (distance >= 0) {
                    Results[index] = new CandidateInfo { Index = sy * Width + sx, Distance = distance };
                } else {
                    Results[index] = new CandidateInfo { Index = -1, Distance = float.MaxValue };
                }
            }
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsPatchFullyKnownBurstStatic(int cx, int cy, NativeParallelHashSet<int> maskSet, int width, int height) {
            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int nx = cx + dx;
                    int ny = cy + dy;

                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        return false;
                    }

                    if (maskSet.Contains(ny * width + nx)) {
                        return false;
                    }
                }
            }
            return true;
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CalculateTextureDistanceBurstStatic(
            int targetX, int targetY, int sourceX, int sourceY,
            NativeParallelHashSet<int> maskSet,
            NativeArray<Color32> pixels,
            NativeArray<float> gaussianWeights,
            int width, int height) {

            float sumDistance = 0f;
            float sumWeight = 0f;
            int weightIdx = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= width || ty < 0 || ty >= height) {
                        weightIdx++;
                        continue;
                    }

                    int targetIdx = ty * width + tx;
                    if (maskSet.Contains(targetIdx)) {
                        weightIdx++;
                        continue;
                    }

                    int sx = sourceX + dx;
                    int sy = sourceY + dy;
                    int sourceIdx = sy * width + sx;

                    float weight = gaussianWeights[weightIdx++];

                    Color32 targetColor = pixels[targetIdx];
                    Color32 sourceColor = pixels[sourceIdx];

                    float dr = targetColor.r - sourceColor.r;
                    float dg = targetColor.g - sourceColor.g;
                    float db = targetColor.b - sourceColor.b;

                    float colorDistSq = dr * dr + dg * dg + db * db;

                    sumDistance += weight * colorDistSq;
                    sumWeight += weight;
                }
            }

            if (sumWeight < 1e-6f) {
                return -1f;
            }

            return sumDistance / sumWeight;
        }

        private static void SearchFullImageBurst(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            ref NativeList<CandidateInfo> candidates,
            NativeArray<float> gaussianWeights,
            int width, int height) {

            int searchStartX = PatchRadius;
            int searchStartY = PatchRadius;
            int searchWidth = width - 2 * PatchRadius;
            int searchHeight = height - 2 * PatchRadius;
            int totalSearchPixels = searchWidth * searchHeight;

            var results = new NativeArray<CandidateInfo>(totalSearchPixels, Allocator.TempJob);

            var job = new SearchFullImageJob {
                TargetX = targetX,
                TargetY = targetY,
                Width = width,
                Height = height,
                SearchWidth = searchWidth,
                SearchStartX = searchStartX,
                SearchStartY = searchStartY,
                MaskSet = maskSet,
                Pixels = pixels,
                GaussianWeights = gaussianWeights,
                Results = results
            };

            // Run in parallel with batch size of 64 for good cache utilization
            job.Schedule(totalSearchPixels, 64).Complete();

            // Collect valid results
            for (int i = 0; i < totalSearchPixels; i++) {
                if (results[i].Index >= 0) {
                    candidates.Add(results[i]);
                }
            }

            results.Dispose();
        }

        private static void SortCandidates(ref NativeList<CandidateInfo> candidates) {
            int n = candidates.Length;

            // Only find the K smallest using partial selection in the sort, stop sort afterwards
            for (int i = 0; i < K; i++) {
                int minIdx = i;
                float minDist = candidates[i].Distance;

                for (int j = i + 1; j < n; j++) {
                    if (candidates[j].Distance < minDist) {
                        minDist = candidates[j].Distance;
                        minIdx = j;
                    }
                }

                // Swap minimum to position i
                if (minIdx != i) {
                    (candidates[i], candidates[minIdx]) = (candidates[minIdx], candidates[i]);
                }
            }
        }

        [BurstCompile]
        private static void FillPatchWithTrimmedMeanBurst(
            int targetX, int targetY,
            ref NativeList<int> candidatePatches,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            int numCandidates = candidatePatches.Length;
            int trimCount = math.max(1, (int)math.round(Alpha * numCandidates));

            // Stack-allocated arrays for small K
            var rValues = new NativeArray<float>(numCandidates, Allocator.Temp);
            var gValues = new NativeArray<float>(numCandidates, Allocator.Temp);
            var bValues = new NativeArray<float>(numCandidates, Allocator.Temp);

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

                    int targetIdx = ty * width + tx;

                    if (!maskSet.Contains(targetIdx)) continue;

                    for (int i = 0; i < numCandidates; i++) {
                        int candidateCenter = candidatePatches[i];
                        int cx = candidateCenter % width;
                        int cy = candidateCenter / width;

                        int sourceX = cx + dx;
                        int sourceY = cy + dy;
                        int sourceIdx = sourceY * width + sourceX;

                        Color32 sourceColor = pixels[sourceIdx];
                        rValues[i] = sourceColor.r;
                        gValues[i] = sourceColor.g;
                        bValues[i] = sourceColor.b;
                    }

                    float r = AlphaTrimmedMeanBurst(ref rValues, trimCount);
                    float g = AlphaTrimmedMeanBurst(ref gValues, trimCount);
                    float b = AlphaTrimmedMeanBurst(ref bValues, trimCount);

                    pixels[targetIdx] = new Color32(
                        (byte)math.clamp(r, 0, 255),
                        (byte)math.clamp(g, 0, 255),
                        (byte)math.clamp(b, 0, 255),
                        255
                    );
                }
            }

            rValues.Dispose();
            gValues.Dispose();
            bValues.Dispose();
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float AlphaTrimmedMeanBurst(ref NativeArray<float> values, int trimCount) {
            int n = values.Length;

            switch (n) {
                case 0:
                    return 0;
                case 1:
                    return values[0];
            }

            // Sort using insertion sort (efficient for small arrays)
            for (int i = 1; i < n; i++) {
                float key = values[i];
                int j = i - 1;
                while (j >= 0 && values[j] > key) {
                    values[j + 1] = values[j];
                    j--;
                }
                values[j + 1] = key;
            }

            int start = math.min(trimCount, n / 2);
            int end = math.max(n - trimCount, n / 2 + 1);

            float sum = 0f;
            int count = 0;

            for (int i = start; i < end; i++) {
                sum += values[i];
                count++;
            }

            return count > 0 ? sum / count : values[n / 2];
        }

        [BurstCompile]
        private static void FillPatchFromNearestBurst(
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
                    if (dist >= nearestDist) continue;
                    
                    nearestDist = dist;
                    nearestIdx = idx;
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
        private static void UpdateConfidenceBurst(
            int targetX, int targetY,
            ref NativeArray<float> confidence,
            ref NativeParallelHashSet<int> maskSet,
            int width, int height) {

            float patchConfidence = CalculateConfidenceBurst(targetX, targetY, ref confidence, width, height);

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
