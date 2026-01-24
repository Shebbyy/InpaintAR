using System;
using System.Collections.Generic;
using InpaintAR.Scripts.Util;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // Nonlocal Texture Matching (NLTM) Inpainting Algorithm
    // See DOI 10.1109/TIP.2018.2880681
    // Extended with PatchMatch-inspired temporal coherence for real-time performance
    public class NltmAlgorithm : AbstractInpaintingAlgorithm {
        private const int PatchRadius = 4; // Radius for Pixel Patches
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

        // Confidence values for each pixel (updated during inpainting)
        private float[] m_confidence;

        // Precomputed Gaussian weights for patch matching
        private float[] m_gaussianWeights;

        // Temporal cache: stores best candidate positions from previous frame
        // Key: approximate target region, Value: list of good candidate positions
        private List<int> m_cachedCandidates = new();
        private bool m_hasCachedCandidates;

        // Random number generator for sparse sampling
        private readonly System.Random m_random = new();
        private int m_prevMaskLength = -1;

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            m_width = TextureUtility.GetImageWidth(source);
            m_height = TextureUtility.GetImageHeight(source);
            m_pixelCount = m_width * m_height;

            // if mask changed -> invalidate cache
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

            // Initialize confidence map: 1 for known pixels, 0 for unknown
            InitializeConfidence(maskPixelIndices);

            // Precompute Gaussian weights for patch matching
            PrecomputeGaussianWeights();

            // Clear candidates found this frame (will be populated during inpainting)
            List<int> newCandidates = new();

            InpaintNltm(maskPixelIndices, newCandidates);

            // Update cache for next frame
            if (newCandidates.Count > 0) {
                m_cachedCandidates = newCandidates;
                m_hasCachedCandidates = true;
            }

            Texture2D resultImage = new Texture2D(m_width, m_height, TextureFormat.RGBA32, false);
            resultImage.SetPixels32(MInpaintedPixelBuffer);
            resultImage.Apply();

            return resultImage;
        }

        // Call this to invalidate cache (e.g., on scene change)
        public void InvalidateCache() {
            m_hasCachedCandidates = false;
            m_cachedCandidates.Clear();
        }

        private void InitializeConfidence(HashSet<int> maskPixelIndices) {
            m_confidence = new float[m_pixelCount];
            for (int i = 0; i < m_pixelCount; i++) {
                m_confidence[i] = maskPixelIndices.Contains(i) ? 0f : 1f;
            }
        }

        private void PrecomputeGaussianWeights() {
            m_gaussianWeights = new float[PatchArea];
            float sigma2 = 2f * GaussianSigma * GaussianSigma;
            int idx = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    float distSq = dx * dx + dy * dy;
                    m_gaussianWeights[idx++] = Mathf.Exp(-distSq / sigma2);
                }
            }
        }

        private void InpaintNltm(HashSet<int> maskPixelIndices, List<int> outNewCandidates) {
            while (maskPixelIndices.Count > 0) {
                int targetIndex = GetTargetPatchIndex(maskPixelIndices);
                if (targetIndex < 0) break;

                int targetX = targetIndex % m_width;
                int targetY = targetIndex / m_width;

                List<int> candidatePatches = FindCandidatePatches(targetX, targetY, maskPixelIndices);

                if (candidatePatches.Count == 0) {
                    FillPatchFromNearest(targetX, targetY, maskPixelIndices);
                } else {
                    FillPatchWithTrimmedMean(targetX, targetY, candidatePatches, maskPixelIndices);

                    // Store candidates for next frame's cache
                    foreach (int candidate in candidatePatches) {
                        if (!outNewCandidates.Contains(candidate)) {
                            outNewCandidates.Add(candidate);
                        }
                    }
                }

                UpdateConfidence(targetX, targetY, maskPixelIndices);
            }
        }

        private int GetTargetPatchIndex(HashSet<int> maskPixelIndices) {
            HashSet<int> contourPixels = GetContourPixels(maskPixelIndices);
            float maxPriority = -1f;
            int bestIndex = -1;

            foreach (int contourPixel in contourPixels) {
                int x = contourPixel % m_width;
                int y = contourPixel / m_width;

                float confidence = CalculateConfidence(x, y);
                Vector2 normal = EstimateBoundaryNormal(x, y, maskPixelIndices);
                float data = CalculateDataTerm(x, y, normal, maskPixelIndices);

                float priority = confidence * data;

                if (priority <= maxPriority) continue;
                maxPriority = priority;
                bestIndex = contourPixel;
            }

            return bestIndex;
        }

        private HashSet<int> GetContourPixels(HashSet<int> maskPixelIndices) {
            HashSet<int> contourPixels = new HashSet<int>();
            int[] directions = { -1, 1, -m_width, m_width };

            foreach (int index in maskPixelIndices) {
                int x = index % m_width;

                foreach (int dir in directions) {
                    int neighbor = index + dir;

                    if (neighbor < 0 || neighbor >= m_pixelCount) continue;

                    switch (dir) {
                        case -1 when x == 0:
                        case 1 when x == m_width - 1:
                            continue;
                    }

                    if (maskPixelIndices.Contains(neighbor)) continue;
                    contourPixels.Add(index);
                    break;
                }
            }

            return contourPixels;
        }

        private float CalculateConfidence(int px, int py) {
            float sumConfidence = 0f;
            int validPixels = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int nx = px + dx;
                    int ny = py + dy;

                    if (nx < 0 || nx >= m_width || ny < 0 || ny >= m_height) continue;

                    int idx = ny * m_width + nx;
                    sumConfidence += m_confidence[idx];
                    validPixels++;
                }
            }

            return validPixels > 0 ? sumConfidence / validPixels : 0f;
        }

        private Vector2 EstimateBoundaryNormal(int px, int py, HashSet<int> maskPixelIndices) {
            float GetMaskVal(int x, int y) {
                if (x < 0 || x >= m_width || y < 0 || y >= m_height) return 0f;
                return maskPixelIndices.Contains(y * m_width + x) ? 1f : 0f;
            }

            float dx = (GetMaskVal(px + 1, py) - GetMaskVal(px - 1, py)) / 2f;
            float dy = (GetMaskVal(px, py + 1) - GetMaskVal(px, py - 1)) / 2f;

            Vector2 normal = new Vector2(dx, dy);
            return normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector2.up;
        }

        private float CalculateDataTerm(int px, int py, Vector2 normal, HashSet<int> maskPixelIndices) {
            Vector2 gradient = ComputeGradient(px, py, maskPixelIndices);
            Vector2 isophote = new Vector2(-gradient.y, gradient.x);
            float dot = Mathf.Abs(isophote.x * normal.x + isophote.y * normal.y);
            return dot / AlphaNorm + 0.001f;
        }

        private Vector2 ComputeGradient(int px, int py, HashSet<int> maskPixelIndices) {
            int centerIdx = py * m_width + px;

            float GetVal(int x, int y) {
                if (x < 0 || x >= m_width || y < 0 || y >= m_height) {
                    return ((Color)MInpaintedPixelBuffer[centerIdx]).grayscale;
                }
                int idx = y * m_width + x;
                if (maskPixelIndices.Contains(idx)) {
                    return ((Color)MInpaintedPixelBuffer[centerIdx]).grayscale;
                }
                return ((Color)MInpaintedPixelBuffer[idx]).grayscale;
            }

            float gx = 0f, gy = 0f;

            gx += -1f * GetVal(px - 1, py - 1) + 1f * GetVal(px + 1, py - 1);
            gx += -2f * GetVal(px - 1, py) + 2f * GetVal(px + 1, py);
            gx += -1f * GetVal(px - 1, py + 1) + 1f * GetVal(px + 1, py + 1);

            gy += 1f * GetVal(px - 1, py - 1) + 2f * GetVal(px, py - 1) + 1f * GetVal(px + 1, py - 1);
            gy += -1f * GetVal(px - 1, py + 1) - 2f * GetVal(px, py + 1) - 1f * GetVal(px + 1, py + 1);

            return new Vector2(gx, gy);
        }

        // <summary>
        // Find K best matching candidate patches using temporal caching.
        // First frame: full global search
        // Subsequent frames: local search around cached positions + sparse random sampling
        // </summary>
        private List<int> FindCandidatePatches(int targetX, int targetY, HashSet<int> maskPixelIndices) {
            List<(int index, float distance)> candidates = new();

            if (m_hasCachedCandidates && m_cachedCandidates.Count > 0) {
                // Local search around cached candidate positions
                SearchAroundCachedCandidates(targetX, targetY, maskPixelIndices, candidates);

                // Add sparse random samples to discover new good matches
                SearchRandomSamples(targetX, targetY, maskPixelIndices, candidates);
            } else {
                // First frame: full global search
                SearchFullImage(targetX, targetY, maskPixelIndices, candidates);
            }

            // Sort by distance and take K best
            candidates.Sort((a, b) => a.distance.CompareTo(b.distance));

            List<int> result = new List<int>();
            int count = Mathf.Min(K, candidates.Count);
            for (int i = 0; i < count; i++) {
                result.Add(candidates[i].index);
            }

            return result;
        }

        // Search locally around each cached candidate position
        private void SearchAroundCachedCandidates(int targetX, int targetY, HashSet<int> maskPixelIndices,
            List<(int index, float distance)> candidates) {
            HashSet<int> searched = new HashSet<int>();

            foreach (int cachedPos in m_cachedCandidates) {
                int cachedX = cachedPos % m_width;
                int cachedY = cachedPos / m_width;

                // Search in local window around cached position
                int minX = Mathf.Max(PatchRadius, cachedX - LocalSearchRadius);
                int maxX = Mathf.Min(m_width - PatchRadius - 1, cachedX + LocalSearchRadius);
                int minY = Mathf.Max(PatchRadius, cachedY - LocalSearchRadius);
                int maxY = Mathf.Min(m_height - PatchRadius - 1, cachedY + LocalSearchRadius);

                for (int sy = minY; sy <= maxY; sy++) {
                    for (int sx = minX; sx <= maxX; sx++) {
                        int idx = sy * m_width + sx;

                        // Skip already searched positions
                        if (!searched.Add(idx)) continue;

                        // Skip if overlaps with target
                        if (Mathf.Abs(sx - targetX) <= PatchRadius && Mathf.Abs(sy - targetY) <= PatchRadius) {
                            continue;
                        }

                        if (!IsPatchFullyKnown(sx, sy, maskPixelIndices)) continue;

                        float distance = CalculateTextureDistance(targetX, targetY, sx, sy, maskPixelIndices);
                        if (distance >= 0) {
                            candidates.Add((idx, distance));
                        }
                    }
                }
            }
        }

        // Add sparse random samples across the image to discover new good matches
        private void SearchRandomSamples(int targetX, int targetY, HashSet<int> maskPixelIndices,
            List<(int index, float distance)> candidates) {
            for (int i = 0; i < RandomSampleCount; i++) {
                int sx = m_random.Next(PatchRadius, m_width - PatchRadius);
                int sy = m_random.Next(PatchRadius, m_height - PatchRadius);

                // Skip if overlaps with target
                if (Mathf.Abs(sx - targetX) <= PatchRadius && Mathf.Abs(sy - targetY) <= PatchRadius) {
                    continue;
                }

                if (!IsPatchFullyKnown(sx, sy, maskPixelIndices)) continue;

                float distance = CalculateTextureDistance(targetX, targetY, sx, sy, maskPixelIndices);
                if (distance >= 0) {
                    candidates.Add((sy * m_width + sx, distance));
                }
            }
        }

        // Full global search (used on first frame when no cache exists)
        private void SearchFullImage(int targetX, int targetY, HashSet<int> maskPixelIndices,
            List<(int index, float distance)> candidates) {
            for (int sy = PatchRadius; sy < m_height - PatchRadius; sy++) {
                for (int sx = PatchRadius; sx < m_width - PatchRadius; sx++) {
                    // Skip if overlaps with target
                    if (Mathf.Abs(sx - targetX) <= PatchRadius && Mathf.Abs(sy - targetY) <= PatchRadius) {
                        continue;
                    }

                    if (!IsPatchFullyKnown(sx, sy, maskPixelIndices)) continue;

                    float distance = CalculateTextureDistance(targetX, targetY, sx, sy, maskPixelIndices);
                    if (distance >= 0) {
                        candidates.Add((sy * m_width + sx, distance));
                    }
                }
            }
        }

        private bool IsPatchFullyKnown(int cx, int cy, HashSet<int> maskPixelIndices) {
            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int nx = cx + dx;
                    int ny = cy + dy;

                    if (nx < 0 || nx >= m_width || ny < 0 || ny >= m_height) {
                        return false;
                    }

                    if (maskPixelIndices.Contains(ny * m_width + nx)) {
                        return false;
                    }
                }
            }
            return true;
        }

        private float CalculateTextureDistance(int targetX, int targetY, int sourceX, int sourceY,
            HashSet<int> maskPixelIndices) {
            float sumDistance = 0f;
            float sumWeight = 0f;
            int weightIdx = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= m_width || ty < 0 || ty >= m_height) {
                        weightIdx++;
                        continue;
                    }

                    int targetIdx = ty * m_width + tx;
                    if (maskPixelIndices.Contains(targetIdx)) {
                        weightIdx++;
                        continue;
                    }

                    int sx = sourceX + dx;
                    int sy = sourceY + dy;
                    int sourceIdx = sy * m_width + sx;

                    float weight = m_gaussianWeights[weightIdx++];

                    Color32 targetColor = MInpaintedPixelBuffer[targetIdx];
                    Color32 sourceColor = MInpaintedPixelBuffer[sourceIdx];

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

        private void FillPatchWithTrimmedMean(int targetX, int targetY, List<int> candidatePatches,
            HashSet<int> maskPixelIndices) {
            int numCandidates = candidatePatches.Count;
            int trimCount = Mathf.Max(1, Mathf.RoundToInt(Alpha * numCandidates));

            float[] rValues = new float[numCandidates];
            float[] gValues = new float[numCandidates];
            float[] bValues = new float[numCandidates];

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= m_width || ty < 0 || ty >= m_height) continue;

                    int targetIdx = ty * m_width + tx;

                    if (!maskPixelIndices.Contains(targetIdx)) continue;

                    for (int i = 0; i < numCandidates; i++) {
                        int candidateCenter = candidatePatches[i];
                        int cx = candidateCenter % m_width;
                        int cy = candidateCenter / m_width;

                        int sourceX = cx + dx;
                        int sourceY = cy + dy;
                        int sourceIdx = sourceY * m_width + sourceX;

                        Color32 sourceColor = MInpaintedPixelBuffer[sourceIdx];
                        rValues[i] = sourceColor.r;
                        gValues[i] = sourceColor.g;
                        bValues[i] = sourceColor.b;
                    }

                    float r = AlphaTrimmedMean(rValues, trimCount);
                    float g = AlphaTrimmedMean(gValues, trimCount);
                    float b = AlphaTrimmedMean(bValues, trimCount);

                    MInpaintedPixelBuffer[targetIdx] = new Color32(
                        (byte)Mathf.Clamp(r, 0, 255),
                        (byte)Mathf.Clamp(g, 0, 255),
                        (byte)Mathf.Clamp(b, 0, 255),
                        255
                    );
                }
            }
        }

        private static float AlphaTrimmedMean(float[] values, int trimCount) {
            int n = values.Length;

            if (n == 0) return 0;
            if (n == 1) return values[0];

            Array.Sort(values);

            int start = Mathf.Min(trimCount, n / 2);
            int end = Mathf.Max(n - trimCount, n / 2 + 1);

            float sum = 0f;
            int count = 0;

            for (int i = start; i < end; i++) {
                sum += values[i];
                count++;
            }

            return count > 0 ? sum / count : values[n / 2];
        }

        private void FillPatchFromNearest(int targetX, int targetY, HashSet<int> maskPixelIndices) {
            int nearestIdx = -1;
            float nearestDist = float.MaxValue;

            for (int dy = -PatchRadius * 2; dy <= PatchRadius * 2; dy++) {
                for (int dx = -PatchRadius * 2; dx <= PatchRadius * 2; dx++) {
                    int nx = targetX + dx;
                    int ny = targetY + dy;

                    if (nx < 0 || nx >= m_width || ny < 0 || ny >= m_height) continue;

                    int idx = ny * m_width + nx;
                    if (maskPixelIndices.Contains(idx)) continue;

                    float dist = dx * dx + dy * dy;
                    if (dist >= nearestDist) continue;

                    nearestDist = dist;
                    nearestIdx = idx;
                }
            }

            if (nearestIdx < 0) return;

            Color32 fillColor = MInpaintedPixelBuffer[nearestIdx];

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= m_width || ty < 0 || ty >= m_height) continue;

                    int targetIdx = ty * m_width + tx;
                    if (!maskPixelIndices.Contains(targetIdx)) continue;

                    MInpaintedPixelBuffer[targetIdx] = fillColor;
                }
            }
        }

        private void UpdateConfidence(int targetX, int targetY, HashSet<int> maskPixelIndices) {
            float patchConfidence = CalculateConfidence(targetX, targetY);

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= m_width || ty < 0 || ty >= m_height) continue;

                    int targetIdx = ty * m_width + tx;

                    // Update confidence for filled pixels
                    if (!maskPixelIndices.Contains(targetIdx)) continue;

                    // Confidence of filled pixel is set to patch confidence
                    m_confidence[targetIdx] = patchConfidence;

                    // Remove from mask
                    maskPixelIndices.Remove(targetIdx);
                }
            }
        }
    }
}
