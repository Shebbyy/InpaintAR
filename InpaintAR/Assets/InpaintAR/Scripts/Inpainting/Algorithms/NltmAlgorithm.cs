using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using InpaintAR.Scripts.Util;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = Unity.Mathematics.Random;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // Nonlocal Texture Matching (NLTM) Inpainting Algorithm
    // See DOI 10.1109/TIP.2018.2880681
    // Extended with PatchMatch-inspired temporal coherence for real-time performance
    public class NltmAlgorithm : AbstractInpaintingAlgorithm {
        private const int PatchRadius = 4;
        private const int PatchSize = 2 * PatchRadius + 1;
        private const int PatchArea = PatchSize * PatchSize;

        // Number of candidate patches K (5-10 recommended according to paper)
        private const int K = 5;

        // Alpha for trimmed mean filter (10-20% recommended by paper to be sliced from each end)
        private const float Alpha = 0.15f;

        // Gaussian sigma for texture matching weight
        private const float GaussianSigma = PatchRadius / 2.0f;

        // Normalization constant for data term
        private const float AlphaNorm = 255.0f;

        // Local search radius around cached candidates (PatchMatch-inspired)
        private const int LocalSearchRadius = 10;

        // Number of random samples for discovering new candidates
        private const int RandomSampleCount = 10;

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
            m_width = TextureUtility.GetImageWidth(source);
            m_height = TextureUtility.GetImageHeight(source);
            m_pixelCount = m_width * m_height;

            // If mask changed, invalidate cache
            int curCount = maskPixelIndices.Count;
            if (curCount != m_prevMaskLength) {
                InvalidateCache();
                m_prevMaskLength = curCount;
            }

            // Initialize output buffer
            if (MInpaintedPixelBuffer == null || MInpaintedPixelBuffer.Length != m_pixelCount) {
                MInpaintedPixelBuffer = new Color32[m_pixelCount];
            }
            Array.Copy(MSourcePixelBuffer, MInpaintedPixelBuffer, m_pixelCount);

            // Precompute Gaussian weights (only once)
            if (!m_gaussianWeightsInitialized) {
                PrecomputeGaussianWeights();
            }

            // Convert to native containers for Burst
            var pixels = new NativeArray<Color32>(MInpaintedPixelBuffer, Allocator.TempJob);
            var confidence = new NativeArray<float>(m_pixelCount, Allocator.TempJob);
            var maskSet = new NativeParallelHashSet<int>(maskPixelIndices.Count, Allocator.TempJob);

            // Initialize confidence: 1 for known, 0 for unknown
            var initJob = new InitializeConfidenceJob {
                Confidence = confidence,
                MaskSet = maskSet
            };
            initJob.Schedule(m_pixelCount, 64).Complete();

            // Copy mask indices to hash set
            foreach (int idx in maskPixelIndices) {
                maskSet.Add(idx);
            }

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

            // Copy results back
            pixels.CopyTo(MInpaintedPixelBuffer);

            // Dispose native containers
            pixels.Dispose();
            confidence.Dispose();
            maskSet.Dispose();
            cachedCandidates.Dispose();
            newCandidates.Dispose();

            // Create result texture
            Texture2D resultImage = new Texture2D(m_width, m_height, TextureFormat.RGBA32, false);
            resultImage.SetPixels32(MInpaintedPixelBuffer);
            resultImage.Apply();

            return resultImage;
        }

        private void InvalidateCache() {
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

            var random = new Random((uint)DateTime.Now.Ticks);
            var contourPixels = new NativeList<int>(Allocator.Temp);
            var candidates = new NativeList<CandidateInfo>(Allocator.Temp);
            var searched = new NativeParallelHashSet<int>(1024, Allocator.Temp);
            var candidatePatches = new NativeList<int>(K, Allocator.Temp);

            while (maskSet.Any()) {
                // Step 1: Find contour pixels
                contourPixels.Clear();
                GetContourPixelsBurst(ref maskSet, ref contourPixels, m_width, m_height);

                if (contourPixels.Length == 0) break;

                // Step 2: Find target patch with highest priority
                int targetIndex = GetTargetPatchIndexBurst(
                    ref contourPixels, ref confidence, ref maskSet, ref pixels,
                    m_width, m_height);

                if (targetIndex < 0) break;

                int targetX = targetIndex % m_width;
                int targetY = targetIndex / m_width;

                // Step 3: Find K best candidate patches
                candidates.Clear();
                searched.Clear();
                candidatePatches.Clear();

                bool useCached = cachedCandidates.Length > 0;
                if (useCached) {
                    SearchAroundCachedCandidatesBurst(
                        targetX, targetY, ref maskSet, ref pixels, ref cachedCandidates,
                        ref candidates, ref searched, m_gaussianWeights, m_width, m_height);

                    SearchRandomSamplesBurst(
                        targetX, targetY, ref maskSet, ref pixels, ref candidates,
                        ref random, m_gaussianWeights, m_width, m_height);
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
            searched.Dispose();
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

        [BurstCompile]
        private static void GetContourPixelsBurst(
            ref NativeParallelHashSet<int> maskSet,
            ref NativeList<int> contourPixels,
            int width, int height) {

            int pixelCount = width * height;
            var maskArray = maskSet.ToNativeArray(Allocator.Temp);

            foreach (var index in maskArray) {
                int x = index % width;

                // Check 4-connected neighbors
                bool isContour = false;

                // Left
                if (x > 0) {
                    int neighbor = index - 1;
                    if (!maskSet.Contains(neighbor)) isContour = true;
                }
                // Right
                if (!isContour && x < width - 1) {
                    int neighbor = index + 1;
                    if (!maskSet.Contains(neighbor)) isContour = true;
                }
                // Up
                if (!isContour && index >= width) {
                    int neighbor = index - width;
                    if (!maskSet.Contains(neighbor)) isContour = true;
                }
                // Down
                if (!isContour && index < pixelCount - width) {
                    int neighbor = index + width;
                    if (!maskSet.Contains(neighbor)) isContour = true;
                }

                if (isContour) {
                    contourPixels.Add(index);
                }
            }

            maskArray.Dispose();
        }

        [BurstCompile]
        private static int GetTargetPatchIndexBurst(
            ref NativeList<int> contourPixels,
            ref NativeArray<float> confidence,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            float maxPriority = -1f;
            int bestIndex = -1;

            foreach (var contourPixel in contourPixels) {
                int x = contourPixel % width;
                int y = contourPixel / width;

                float conf = CalculateConfidenceBurst(x, y, ref confidence, width, height);
                float2 normal = EstimateBoundaryNormalBurst(x, y, ref maskSet, width, height);
                float data = CalculateDataTermBurst(x, y, normal, ref maskSet, ref pixels, width, height);

                float priority = conf * data;

                if (priority <= maxPriority) continue;
                
                maxPriority = priority;
                bestIndex = contourPixel;
            }

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

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float2 EstimateBoundaryNormalBurst(int px, int py, ref NativeParallelHashSet<int> maskSet, int width, int height) {
            var set = maskSet;

            float GetMaskVal(int x, int y) {
                if (x < 0 || x >= width || y < 0 || y >= height) return 0f;
                return set.Contains(y * width + x) ? 1f : 0f;
            }

            float dx = (GetMaskVal(px + 1, py) - GetMaskVal(px - 1, py)) / 2f;
            float dy = (GetMaskVal(px, py + 1) - GetMaskVal(px, py - 1)) / 2f;

            float2 normal = new float2(dx, dy);
            float sqrMag = normal.x * normal.x + normal.y * normal.y;

            return sqrMag > 1e-6f 
                ? math.normalize(normal) 
                : new float2(0f, 1f);
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CalculateDataTermBurst(
            int px, int py, float2 normal,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            float2 gradient = ComputeGradientBurst(px, py, ref maskSet, ref pixels, width, height);
            float2 isophote = new float2(-gradient.y, gradient.x);
            float dot = math.abs(isophote.x * normal.x + isophote.y * normal.y);
            return dot / AlphaNorm + 0.001f;
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float2 ComputeGradientBurst(
            int px, int py,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            int centerIdx = py * width + px;
            float centerGray = GetGrayscale(pixels[centerIdx]);

            var set = maskSet;

            var array = pixels;

            float GetVal(int x, int y) {
                if (x < 0 || x >= width || y < 0 || y >= height) {
                    return centerGray;
                }
                int idx = y * width + x;
                return set.Contains(idx) ? centerGray : GetGrayscale(array[idx]);
            }

            float gx = 0f, gy = 0f;

            gx += -1f * GetVal(px - 1, py - 1) + 1f * GetVal(px + 1, py - 1);
            gx += -2f * GetVal(px - 1, py) + 2f * GetVal(px + 1, py);
            gx += -1f * GetVal(px - 1, py + 1) + 1f * GetVal(px + 1, py + 1);

            gy += 1f * GetVal(px - 1, py - 1) + 2f * GetVal(px, py - 1) + 1f * GetVal(px + 1, py - 1);
            gy += -1f * GetVal(px - 1, py + 1) - 2f * GetVal(px, py + 1) - 1f * GetVal(px + 1, py + 1);

            return new float2(gx, gy);
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float GetGrayscale(Color32 c) {
            return (c.r * 0.299f + c.g * 0.587f + c.b * 0.114f) / 255f;
        }

        [BurstCompile]
        private static void SearchAroundCachedCandidatesBurst(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            ref NativeList<int> cachedCandidates,
            ref NativeList<CandidateInfo> candidates,
            ref NativeParallelHashSet<int> searched,
            NativeArray<float> gaussianWeights,
            int width, int height) {
            foreach (var cachedPos in cachedCandidates) {
                int cachedX = cachedPos % width;
                int cachedY = cachedPos / width;

                int minX = math.max(PatchRadius, cachedX - LocalSearchRadius);
                int maxX = math.min(width - PatchRadius - 1, cachedX + LocalSearchRadius);
                int minY = math.max(PatchRadius, cachedY - LocalSearchRadius);
                int maxY = math.min(height - PatchRadius - 1, cachedY + LocalSearchRadius);

                for (int sy = minY; sy <= maxY; sy++) {
                    for (int sx = minX; sx <= maxX; sx++) {
                        int idx = sy * width + sx;

                        if (searched.Contains(idx)) continue;
                        searched.Add(idx);

                        if (math.abs(sx - targetX) <= PatchRadius && math.abs(sy - targetY) <= PatchRadius) {
                            continue;
                        }

                        if (!IsPatchFullyKnownBurst(sx, sy, ref maskSet, width, height)) continue;

                        float distance = CalculateTextureDistanceBurst(
                            targetX, targetY, sx, sy, ref maskSet, ref pixels, gaussianWeights, width, height);

                        if (distance >= 0) {
                            candidates.Add(new CandidateInfo { Index = idx, Distance = distance });
                        }
                    }
                }
            }
        }

        [BurstCompile]
        private static void SearchRandomSamplesBurst(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            ref NativeList<CandidateInfo> candidates,
            ref Random random,
            NativeArray<float> gaussianWeights,
            int width, int height) {

            for (int i = 0; i < RandomSampleCount; i++) {
                int sx = random.NextInt(PatchRadius, width - PatchRadius);
                int sy = random.NextInt(PatchRadius, height - PatchRadius);

                if (math.abs(sx - targetX) <= PatchRadius && math.abs(sy - targetY) <= PatchRadius) {
                    continue;
                }

                if (!IsPatchFullyKnownBurst(sx, sy, ref maskSet, width, height)) continue;

                float distance = CalculateTextureDistanceBurst(
                    targetX, targetY, sx, sy, ref maskSet, ref pixels, gaussianWeights, width, height);

                if (distance >= 0) {
                    candidates.Add(new CandidateInfo { Index = sy * width + sx, Distance = distance });
                }
            }
        }

        [BurstCompile]
        private static void SearchFullImageBurst(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            ref NativeList<CandidateInfo> candidates,
            NativeArray<float> gaussianWeights,
            int width, int height) {

            for (int sy = PatchRadius; sy < height - PatchRadius; sy++) {
                for (int sx = PatchRadius; sx < width - PatchRadius; sx++) {
                    if (math.abs(sx - targetX) <= PatchRadius && math.abs(sy - targetY) <= PatchRadius) {
                        continue;
                    }

                    if (!IsPatchFullyKnownBurst(sx, sy, ref maskSet, width, height)) continue;

                    float distance = CalculateTextureDistanceBurst(
                        targetX, targetY, sx, sy, ref maskSet, ref pixels, gaussianWeights, width, height);

                    if (distance >= 0) {
                        candidates.Add(new CandidateInfo { Index = sy * width + sx, Distance = distance });
                    }
                }
            }
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsPatchFullyKnownBurst(int cx, int cy, ref NativeParallelHashSet<int> maskSet, int width, int height) {
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
        private static float CalculateTextureDistanceBurst(
            int targetX, int targetY, int sourceX, int sourceY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
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

        private static void SortCandidates(ref NativeList<CandidateInfo> candidates) {
            // Simple insertion sort for small arrays (typically K candidates)
            for (int i = 1; i < candidates.Length; i++) {
                var key = candidates[i];
                int j = i - 1;
                while (j >= 0 && candidates[j].Distance > key.Distance) {
                    candidates[j + 1] = candidates[j];
                    j--;
                }
                candidates[j + 1] = key;
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
