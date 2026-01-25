using System;
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

        // Normalization constant for data term
        private const float AlphaNorm = 255.0f;

        // Threshold for switching from geometry phase to texture phase
        // When max data term falls below this, switch to texture phase
        private const float GeometryThreshold = 0.01f;

        private int m_width;
        private int m_height;
        private int m_pixelCount;

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            m_width = TextureUtility.GetImageWidth(source);
            m_height = TextureUtility.GetImageHeight(source);
            m_pixelCount = m_width * m_height;

            var pixels = new NativeArray<Color32>(MPixelBuffer, Allocator.TempJob);
            var confidence = new NativeArray<float>(m_pixelCount, Allocator.TempJob);
            var maskSet = new NativeParallelHashSet<int>(maskPixelIndices.Count, Allocator.TempJob);

            // Initialize confidence: 1 for known, 0 for unknown
            var initJob = new InitializeConfidenceJob {
                Confidence = confidence
            };
            initJob.Schedule(m_pixelCount, 64).Complete();

            // Copy mask indices to hash set and set confidence to 0
            foreach (int idx in maskPixelIndices) {
                maskSet.Add(idx);
                confidence[idx] = 0f;
            }

            // Run ELTM inpainting
            InpaintEltmBurst(ref pixels, ref confidence, ref maskSet);

            pixels.CopyTo(MPixelBuffer);

            pixels.Dispose();
            confidence.Dispose();
            maskSet.Dispose();

            Texture2D resultImage = new Texture2D(m_width, m_height, TextureFormat.RGBA32, false);
            resultImage.SetPixels32(MPixelBuffer);
            resultImage.Apply();

            return resultImage;
        }

        [BurstCompile]
        private struct InitializeConfidenceJob : IJobParallelFor {
            [WriteOnly] public NativeArray<float> Confidence;

            public void Execute(int index) {
                Confidence[index] = 1f;
            }
        }

        private void InpaintEltmBurst(
            ref NativeArray<Color32> pixels,
            ref NativeArray<float> confidence,
            ref NativeParallelHashSet<int> maskSet) {

            var contourPixels = new NativeList<int>(Allocator.Temp);
            bool geometryPhase = true; // Start with geometry phase

            while (!maskSet.IsEmpty) {
                // Step 1: Find contour pixels
                contourPixels.Clear();
                GetContourPixelsBurst(ref maskSet, ref contourPixels, m_width, m_height);

                if (contourPixels.Length == 0) break;

                // Step 2: Find target patch with highest priority
                // Two-phase priority: Phase 1 uses D(p), Phase 2 uses C(p)
                int targetIndex;

                if (geometryPhase) {
                    targetIndex = GetTargetPatchGeometryPhase(
                        ref contourPixels, ref maskSet, ref pixels,
                        m_width, m_height, out var maxDataTerm);

                    // Switch to texture phase when geometry structures are done
                    if (maxDataTerm < GeometryThreshold) {
                        geometryPhase = false;
                    }
                } else {
                    targetIndex = GetTargetPatchTexturePhase(
                        ref contourPixels, ref confidence, m_width, m_height);
                }

                if (targetIndex < 0) break;

                int targetX = targetIndex % m_width;
                int targetY = targetIndex / m_width;

                // Step 3: Find best matching patch in local search window using SSD
                int bestPatch = FindBestMatchingPatch(
                    targetX, targetY, ref maskSet, ref pixels, m_width, m_height);

                // Step 4: Fill target patch from best match
                if (bestPatch >= 0) {
                    FillPatchFromSource(targetX, targetY, bestPatch, ref maskSet, ref pixels, m_width, m_height);
                } else {
                    FillPatchFromNearest(targetX, targetY, ref maskSet, ref pixels, m_width, m_height);
                }

                // Step 5: Update confidence and remove filled pixels from mask
                UpdateConfidence(targetX, targetY, ref confidence, ref maskSet, m_width, m_height);
            }

            contourPixels.Dispose();
        }

        [BurstCompile]
        private static void GetContourPixelsBurst(
            ref NativeParallelHashSet<int> maskSet,
            ref NativeList<int> contourPixels,
            int width, int height) {

            int pixelCount = width * height;
            var maskArray = maskSet.ToNativeArray(Allocator.Temp);

            for (int i = 0; i < maskArray.Length; i++) {
                int index = maskArray[i];
                int x = index % width;

                bool isContour = false;

                // Left
                if (x > 0 && !maskSet.Contains(index - 1)) isContour = true;
                // Right
                if (!isContour && x < width - 1 && !maskSet.Contains(index + 1)) isContour = true;
                // Up
                if (!isContour && index >= width && !maskSet.Contains(index - width)) isContour = true;
                // Down
                if (!isContour && index < pixelCount - width && !maskSet.Contains(index + width)) isContour = true;

                if (isContour) {
                    contourPixels.Add(index);
                }
            }

            maskArray.Dispose();
        }

        // Phase 1: Priority based on data term D(p) only - propagates geometry
        [BurstCompile]
        private static int GetTargetPatchGeometryPhase(
            ref NativeList<int> contourPixels,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height,
            out float maxDataTerm) {

            float maxPriority = -1f;
            int bestIndex = -1;
            maxDataTerm = 0f;

            for (int i = 0; i < contourPixels.Length; i++) {
                int contourPixel = contourPixels[i];
                int x = contourPixel % width;
                int y = contourPixel / width;

                float2 normal = EstimateBoundaryNormal(x, y, ref maskSet, width, height);
                float data = CalculateDataTerm(x, y, normal, ref maskSet, ref pixels, width, height);

                if (data > maxDataTerm) {
                    maxDataTerm = data;
                }

                if (data > maxPriority) {
                    maxPriority = data;
                    bestIndex = contourPixel;
                }
            }

            return bestIndex;
        }

        // Phase 2: Priority based on confidence term C(p) only - synthesizes texture
        [BurstCompile]
        private static int GetTargetPatchTexturePhase(
            ref NativeList<int> contourPixels,
            ref NativeArray<float> confidence,
            int width, int height) {

            float maxPriority = -1f;
            int bestIndex = -1;

            for (int i = 0; i < contourPixels.Length; i++) {
                int contourPixel = contourPixels[i];
                int x = contourPixel % width;
                int y = contourPixel / width;

                float conf = CalculateConfidence(x, y, ref confidence, width, height);

                if (conf > maxPriority) {
                    maxPriority = conf;
                    bestIndex = contourPixel;
                }
            }

            return bestIndex;
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CalculateConfidence(int px, int py, ref NativeArray<float> confidence, int width, int height) {
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
        private static float2 EstimateBoundaryNormal(int px, int py, ref NativeParallelHashSet<int> maskSet, int width, int height) {
            var set = maskSet;

            float GetMaskVal(int x, int y) {
                if (x < 0 || x >= width || y < 0 || y >= height) return 0f;
                return set.Contains(y * width + x) ? 1f : 0f;
            }

            float dx = (GetMaskVal(px + 1, py) - GetMaskVal(px - 1, py)) / 2f;
            float dy = (GetMaskVal(px, py + 1) - GetMaskVal(px, py - 1)) / 2f;

            float2 normal = new float2(dx, dy);
            float sqrMag = normal.x * normal.x + normal.y * normal.y;

            if (sqrMag > 1e-6f) {
                return math.normalize(normal);
            }
            return new float2(0f, 1f);
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CalculateDataTerm(
            int px, int py, float2 normal,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            float2 gradient = ComputeGradient(px, py, ref maskSet, ref pixels, width, height);
            float2 isophote = new float2(-gradient.y, gradient.x);
            float dot = math.abs(isophote.x * normal.x + isophote.y * normal.y);
            return dot / AlphaNorm + 0.001f;
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float2 ComputeGradient(
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
                if (set.Contains(idx)) {
                    return centerGray;
                }
                return GetGrayscale(array[idx]);
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

        // Find best matching patch using SSD within local search window
        [BurstCompile]
        private static int FindBestMatchingPatch(
            int targetX, int targetY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            float bestDistance = float.MaxValue;
            int bestPatch = -1;

            // Local search window (patch-in-patch strategy from paper)
            int minX = math.max(PatchRadius, targetX - SearchRadius);
            int maxX = math.min(width - PatchRadius - 1, targetX + SearchRadius);
            int minY = math.max(PatchRadius, targetY - SearchRadius);
            int maxY = math.min(height - PatchRadius - 1, targetY + SearchRadius);

            for (int sy = minY; sy <= maxY; sy++) {
                for (int sx = minX; sx <= maxX; sx++) {
                    // Skip if source patch overlaps with target
                    if (math.abs(sx - targetX) <= PatchRadius && math.abs(sy - targetY) <= PatchRadius) {
                        continue;
                    }

                    // Check if source patch is fully known
                    if (!IsPatchFullyKnown(sx, sy, ref maskSet, width, height)) continue;

                    // Calculate SSD
                    float distance = CalculateSsd(targetX, targetY, sx, sy, ref maskSet, ref pixels, width, height);

                    if (distance >= 0 && distance < bestDistance) {
                        bestDistance = distance;
                        bestPatch = sy * width + sx;
                    }
                }
            }

            return bestPatch;
        }

        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsPatchFullyKnown(int cx, int cy, ref NativeParallelHashSet<int> maskSet, int width, int height) {
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

        // Simple SSD (Sum of Squared Differences) - only compare known pixels
        [BurstCompile]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float CalculateSsd(
            int targetX, int targetY, int sourceX, int sourceY,
            ref NativeParallelHashSet<int> maskSet,
            ref NativeArray<Color32> pixels,
            int width, int height) {

            float sumDistance = 0f;
            int validPixels = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

                    int targetIdx = ty * width + tx;

                    // Only compare known pixels in target patch
                    if (maskSet.Contains(targetIdx)) continue;

                    int sx = sourceX + dx;
                    int sy = sourceY + dy;
                    int sourceIdx = sy * width + sx;

                    Color32 targetColor = pixels[targetIdx];
                    Color32 sourceColor = pixels[sourceIdx];

                    float dr = targetColor.r - sourceColor.r;
                    float dg = targetColor.g - sourceColor.g;
                    float db = targetColor.b - sourceColor.b;

                    sumDistance += dr * dr + dg * dg + db * db;
                    validPixels++;
                }
            }

            if (validPixels == 0) {
                return -1f;
            }

            return sumDistance / validPixels;
        }

        // Fill target patch by copying from best matching source patch
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

                    // Only fill unknown pixels
                    if (!maskSet.Contains(targetIdx)) continue;

                    int sx = sourceX + dx;
                    int sy = sourceY + dy;
                    int sourceIdx = sy * width + sx;

                    pixels[targetIdx] = pixels[sourceIdx];
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

            float patchConfidence = CalculateConfidence(targetX, targetY, ref confidence, width, height);

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
