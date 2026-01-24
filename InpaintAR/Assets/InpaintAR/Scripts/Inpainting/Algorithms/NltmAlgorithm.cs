using System;
using System.Collections.Generic;
using InpaintAR.Scripts.Util;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // Nonlocal Texture Matching (NLTM) Inpainting Algorithm
    // See DOI 10.1109/TIP.2018.2880681
    public class NltmAlgorithm : AbstractInpaintingAlgorithm {
        private const int PatchRadius = 4; // Radius for Pixel Patches
        private const int PatchSize = 2 * PatchRadius + 1;
        private const int PatchArea = PatchSize * PatchSize;

        // Number of candidate patches K (5-10 recommended according to paper)
        private const int K = 5;

        // Alpha for trimmed mean filter (10-20% recommended by paper to be sliced from each end)
        private const float Alpha = 0.15f;

        // Gaussian sigma for texture matching weight (relative to patch radius)
        private const float GaussianSigma = PatchRadius / 2.0f;

        // Normalization constant for data term
        private const float AlphaNorm = 255.0f;

        private int m_width;
        private int m_height;
        private int m_pixelCount;

        // Confidence values for each pixel (updated during inpainting)
        private float[] m_confidence;

        // Precomputed Gaussian weights for patch matching
        private float[] m_gaussianWeights;

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            m_width = TextureUtility.GetImageWidth(source);
            m_height = TextureUtility.GetImageHeight(source);
            m_pixelCount = m_width * m_height;

            // Initialize output buffer
            if (MInpaintedPixelBuffer == null || MInpaintedPixelBuffer.Length != m_pixelCount) {
                MInpaintedPixelBuffer = new Color32[m_pixelCount];
            }
            Array.Copy(MSourcePixelBuffer, MInpaintedPixelBuffer, m_pixelCount);

            // Initialize confidence map: 1 for known pixels, 0 for unknown
            InitializeConfidence(maskPixelIndices);

            // Precompute Gaussian weights for patch matching
            PrecomputeGaussianWeights();

            // Run the NLTM inpainting algorithm
            InpaintNltm(maskPixelIndices);

            // Create result texture
            Texture2D resultImage = new Texture2D(m_width, m_height, TextureFormat.RGBA32, false);
            resultImage.SetPixels32(MInpaintedPixelBuffer);
            resultImage.Apply();

            return resultImage;
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

        // Main NLTM inpainting loop
        private void InpaintNltm(HashSet<int> maskPixelIndices) {
            while (maskPixelIndices.Count > 0) {
                // Step 1: Find the target patch with highest priority on the fill front
                int targetIndex = GetTargetPatchIndex(maskPixelIndices);
                if (targetIndex < 0) break;

                int targetX = targetIndex % m_width;
                int targetY = targetIndex / m_width;

                // Step 2: Find K best matching candidate patches using texture similarity
                List<int> candidatePatches = FindCandidatePatches(targetX, targetY, maskPixelIndices);

                if (candidatePatches.Count == 0) {
                    // Fallback: if no candidates found, just copy from nearest known pixel
                    FillPatchFromNearest(targetX, targetY, maskPixelIndices);
                } else {
                    // Step 3: Fill target patch using alpha-trimmed mean filter
                    FillPatchWithTrimmedMean(targetX, targetY, candidatePatches, maskPixelIndices);
                }

                // Step 4: Update confidence values for newly filled pixels
                UpdateConfidence(targetX, targetY, maskPixelIndices);
            }
        }

        // Find the patch on the fill front with highest priority P(p) = C(p) * D(p)
        private int GetTargetPatchIndex(HashSet<int> maskPixelIndices) {
            HashSet<int> contourPixels = GetContourPixels(maskPixelIndices);
            float maxPriority = -1f;
            int bestIndex = -1;

            foreach (int contourPixel in contourPixels) {
                int x = contourPixel % m_width;
                int y = contourPixel / m_width;

                // Calculate confidence term C(p)
                float confidence = CalculateConfidence(x, y);

                // Calculate data term D(p)
                Vector2 normal = EstimateBoundaryNormal(x, y, maskPixelIndices);
                float data = CalculateDataTerm(x, y, normal, maskPixelIndices);

                float priority = confidence * data;

                if (priority <= maxPriority) continue;
                maxPriority = priority;
                bestIndex = contourPixel;
            }

            return bestIndex;
        }

        // Get pixels on the boundary of the inpainting region (fill front δΩ)
        private HashSet<int> GetContourPixels(HashSet<int> maskPixelIndices) {
            HashSet<int> contourPixels = new HashSet<int>();
            int[] directions = { -1, 1, -m_width, m_width };

            foreach (int index in maskPixelIndices) {
                int x = index % m_width;

                foreach (int dir in directions) {
                    int neighbor = index + dir;

                    // Check bounds
                    if (neighbor < 0 || neighbor >= m_pixelCount) continue;

                    switch (dir) {
                        // Check horizontal wrap-around
                        case -1 when x == 0:
                        case 1 when x == m_width - 1:
                            continue;
                    }

                    // If neighbor is known (not in mask), this is a boundary pixel
                    if (maskPixelIndices.Contains(neighbor)) continue;
                    contourPixels.Add(index);
                    break;
                }
            }

            return contourPixels;
        }

        // Calculate confidence term C(p) = sum of confidence in patch / patch area
        // Uses stored confidence values (Equation 1 in paper)
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

        // Estimate the unit normal to the fill front at point p
        private Vector2 EstimateBoundaryNormal(int px, int py, HashSet<int> maskPixelIndices) {
            float GetMaskVal(int x, int y) {
                if (x < 0 || x >= m_width || y < 0 || y >= m_height) return 0f;
                return maskPixelIndices.Contains(y * m_width + x) ? 1f : 0f;
            }

            float dx = (GetMaskVal(px + 1, py) - GetMaskVal(px - 1, py)) / 2f;
            float dy = (GetMaskVal(px, py + 1) - GetMaskVal(px, py - 1)) / 2f;

            Vector2 normal = new Vector2(dx, dy);

            return normal.sqrMagnitude > 1e-6f ? normal.normalized : Vector2.up; // Fallback
        }

        // Calculate data term D(p) = |∇I⊥ · n_p| / α
        // Measures how strongly isophotes flow into the boundary (Equation 2 in paper)
        private float CalculateDataTerm(int px, int py, Vector2 normal, HashSet<int> maskPixelIndices) {
            Vector2 gradient = ComputeGradient(px, py, maskPixelIndices);

            // Isophote direction is perpendicular to gradient
            Vector2 isophote = new Vector2(-gradient.y, gradient.x);

            // Dot product with boundary normal
            float dot = Mathf.Abs(isophote.x * normal.x + isophote.y * normal.y);

            return dot / AlphaNorm + 0.001f; // Small constant to avoid zero priority
        }

        // Compute image gradient using Sobel operator (only from known pixels)
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

            // Sobel operator
            float gx = 0f, gy = 0f;

            gx += -1f * GetVal(px - 1, py - 1) + 1f * GetVal(px + 1, py - 1);
            gx += -2f * GetVal(px - 1, py) + 2f * GetVal(px + 1, py);
            gx += -1f * GetVal(px - 1, py + 1) + 1f * GetVal(px + 1, py + 1);

            gy += 1f * GetVal(px - 1, py - 1) + 2f * GetVal(px, py - 1) + 1f * GetVal(px + 1, py - 1);
            gy += -1f * GetVal(px - 1, py + 1) - 2f * GetVal(px, py + 1) - 1f * GetVal(px + 1, py + 1);

            return new Vector2(gx, gy);
        }

        // Find K best matching candidate patches using Gaussian-weighted nonlocal texture similarity
        // (Equation 4 in paper)
        private List<int> FindCandidatePatches(int targetX, int targetY, HashSet<int> maskPixelIndices) {
            List<(int index, float distance)> candidates = new();

            // Define search region
            int searchMinX, searchMaxX, searchMinY, searchMaxY;

            searchMinX = Mathf.Max(PatchRadius, targetX);
            searchMaxX = Mathf.Min(m_width - PatchRadius - 1, targetX);
            searchMinY = Mathf.Max(PatchRadius, targetY);
            searchMaxY = Mathf.Min(m_height - PatchRadius - 1, targetY);

            // Search for candidate patches
            for (int sy = searchMinY; sy <= searchMaxY; sy++) {
                for (int sx = searchMinX; sx <= searchMaxX; sx++) {
                    // Skip if source patch overlaps with target region
                    if (Mathf.Abs(sx - targetX) <= PatchRadius && Mathf.Abs(sy - targetY) <= PatchRadius) {
                        continue;
                    }

                    // Check if source patch is entirely in known region
                    if (!IsPatchFullyKnown(sx, sy, maskPixelIndices)) {
                        continue;
                    }

                    // Calculate Gaussian-weighted texture distance
                    float distance = CalculateTextureDistance(targetX, targetY, sx, sy, maskPixelIndices);

                    if (distance >= 0) {
                        candidates.Add((sy * m_width + sx, distance));
                    }
                }
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

        // Check if a patch centered at (cx, cy) is entirely in the known region
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

        // Calculate Gaussian-weighted texture distance between target and source patches
        // (Equation 3 in paper: only compare known pixels in target patch)
        private float CalculateTextureDistance(int targetX, int targetY, int sourceX, int sourceY,
            HashSet<int> maskPixelIndices) {
            float sumDistance = 0f;
            float sumWeight = 0f;
            int weightIdx = 0;

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    // Only compare known pixels in target patch
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

                    // Get Gaussian weight
                    float weight = m_gaussianWeights[weightIdx++];

                    // Calculate squared color difference
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

            // Need at least some known pixels to compare
            if (sumWeight < 1e-6f) {
                return -1f; // Invalid
            }

            return sumDistance / sumWeight;
        }

        // Fill target patch using alpha-trimmed mean filter on K candidate patches
        // (Equation 5 in paper)
        private void FillPatchWithTrimmedMean(int targetX, int targetY, List<int> candidatePatches,
            HashSet<int> maskPixelIndices) {
            int numCandidates = candidatePatches.Count;
            int trimCount = Mathf.Max(1, Mathf.RoundToInt(Alpha * numCandidates));

            // Pre-allocate arrays for trimmed mean calculation
            float[] rValues = new float[numCandidates];
            float[] gValues = new float[numCandidates];
            float[] bValues = new float[numCandidates];

            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    int tx = targetX + dx;
                    int ty = targetY + dy;

                    if (tx < 0 || tx >= m_width || ty < 0 || ty >= m_height) continue;

                    int targetIdx = ty * m_width + tx;

                    // Only fill unknown pixels
                    if (!maskPixelIndices.Contains(targetIdx)) continue;

                    // Collect pixel values from all candidate patches
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

                    // Apply alpha-trimmed mean
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

        // Calculate alpha-trimmed mean: sort values, trim alpha% from each end, average the rest
        private static float AlphaTrimmedMean(float[] values, int trimCount) {
            int n = values.Length;

            if (n == 0) return 0;
            if (n == 1) return values[0];

            // Sort the values
            Array.Sort(values);

            // Calculate mean of middle values (excluding trimmed ends)
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

        // Fallback: fill patch from nearest known pixel when no candidates found
        private void FillPatchFromNearest(int targetX, int targetY, HashSet<int> maskPixelIndices) {
            // Find nearest known pixel
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

            // Fill the patch with this color
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

        // Update confidence values and remove filled pixels from mask (Equation 2 in paper)
        private void UpdateConfidence(int targetX, int targetY, HashSet<int> maskPixelIndices) {
            // Get the confidence of the patch center before filling
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
