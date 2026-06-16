using System.Collections.Generic;
using System.Linq;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace InpaintAR.Scripts.Benchmarking.Evaluators {
    // Clutter evaluation based on Feature Congestion model from "Measuring Visual Clutter" by Rosenholtz, Li, and Nakano (2007)
    // See DOI  10.1167/7.2.17
    public class ClutterEvaluator {
        private static readonly List<float> ClutterReductionResults = new();
        private static NativeArray<Color32> _nativeOriginal;
        private static NativeArray<Color32> _nativeInpainted;
        private static NativeArray<byte> _nativeMask;
        private static NativeArray<float> _originalClutter ;
        private static NativeArray<float> _inpaintedClutter;
        private static JobHandle? _jobHandle;
        // Evaluates clutter reduction by comparing Feature Congestion clutter
        // between original and inpainted images within the masked region.
        public static void EvaluateClutterReduction(Color32[] originalImage, Color32[] inpaintedImage, int width, int height,
            HashSet<int> maskPixelIndices) {
            if (_jobHandle is not null) {
                if (!_jobHandle.Value.IsCompleted) return;
                
                float origClut = _originalClutter[0];
                float inpClut = _inpaintedClutter[0];
            
                float clutterReduction = inpClut / origClut * 100;

                _nativeOriginal.Dispose();
                _nativeInpainted.Dispose();
                _nativeMask.Dispose();
                _originalClutter.Dispose();
                _inpaintedClutter.Dispose();

                ClutterReductionResults.Add(clutterReduction);
            }

            // Persistent (not TempJob): on the GPU path this is driven by the throttled
            // AsyncGPUReadback callback, so these can live many frames before the next call disposes
            // them - well past TempJob's 4-frame limit. Persistent has no frame lifetime.
            _nativeOriginal  = new NativeArray<Color32>(originalImage, Allocator.Persistent);
            _nativeInpainted = new NativeArray<Color32>(inpaintedImage, Allocator.Persistent);

            _nativeMask = new NativeArray<byte>(width * height, Allocator.Persistent);
            foreach (int index in maskPixelIndices) {
                _nativeMask[index] = 1;
            }

            _originalClutter = new NativeArray<float>(1, Allocator.Persistent);
            _inpaintedClutter = new NativeArray<float>(1, Allocator.Persistent);

            var clutterJob = new ClutterEvaluationJob {
                OriginalPixels = _nativeOriginal,
                InpaintedPixels = _nativeInpainted,
                Mask = _nativeMask,
                Width = width,
                Height = height,
                OriginalClutter = _originalClutter,
                InpaintedClutter = _inpaintedClutter
            };

            _jobHandle = clutterJob.Schedule();
        }

        [BurstCompile]
        private struct ClutterEvaluationJob : IJob {
            [ReadOnly] public NativeArray<Color32> OriginalPixels;
            [ReadOnly] public NativeArray<Color32> InpaintedPixels;
            [ReadOnly] public NativeArray<byte> Mask;
            public int Width;
            public int Height;

            [WriteOnly] public NativeArray<float> OriginalClutter;
            [WriteOnly] public NativeArray<float> InpaintedClutter;

            // Window radius for local feature computation (paper uses ~1 degree visual angle, we approximate with pixels)
            private const int WindowRadius = 3;

            // Weights for combining feature dimensions (from paper)
            private const float ColorWeight = 1.0f;
            private const float ContrastWeight = 1.0f;
            private const float OrientationWeight = 1.0f;

            public void Execute() {
                float origClutter = ComputeFeatureCongestionClutter(OriginalPixels);
                float inpClutter = ComputeFeatureCongestionClutter(InpaintedPixels);

                OriginalClutter[0] = origClutter;
                InpaintedClutter[0] = inpClutter;
            }

            // Computes Feature Congestion clutter as per Rosenholtz et al. 2007
            // Clutter = weighted sum of color congestion, contrast congestion, and orientation congestion
            private float ComputeFeatureCongestionClutter(NativeArray<Color32> pixels) {
                float totalColorCongestion = 0f;
                float totalContrastCongestion = 0f;
                float totalOrientationCongestion = 0f;
                int count = 0;

                // Process only masked region pixels
                for (int y = WindowRadius; y < Height - WindowRadius; y++) {
                    for (int x = WindowRadius; x < Width - WindowRadius; x++) {
                        int idx = y * Width + x;
                        if (Mask[idx] != 1) continue;

                        // Compute local feature congestion for each dimension
                        float colorCong = ComputeLocalColorCongestion(pixels, x, y);
                        float contrastCong = ComputeLocalContrastCongestion(pixels, x, y);
                        float orientCong = ComputeLocalOrientationCongestion(pixels, x, y);

                        totalColorCongestion += colorCong;
                        totalContrastCongestion += contrastCong;
                        totalOrientationCongestion += orientCong;
                        count++;
                    }
                }

                if (count == 0) return 0f;

                // Average congestion across pixels
                float avgColor = totalColorCongestion / count;
                float avgContrast = totalContrastCongestion / count;
                float avgOrientation = totalOrientationCongestion / count;

                // Combined clutter (equation from paper: weighted sum of feature congestions)
                float clutter = ColorWeight * avgColor + ContrastWeight * avgContrast + OrientationWeight * avgOrientation;

                return clutter;
            }

            // Color congestion: volume of local color distribution in CIELab-like space
            // Approximated as sqrt of determinant of covariance matrix
            private float ComputeLocalColorCongestion(NativeArray<Color32> pixels, int cx, int cy) {
                // Collect color values in local window (convert to opponent color space as approximation to Lab)
                float sumL = 0f, sumA = 0f, sumB = 0f;
                float sumLL = 0f, sumAA = 0f, sumBB = 0f;
                float sumLA = 0f, sumLB = 0f, sumAB = 0f;
                int n = 0;

                for (int dy = -WindowRadius; dy <= WindowRadius; dy++) {
                    for (int dx = -WindowRadius; dx <= WindowRadius; dx++) {
                        int x = cx + dx;
                        int y = cy + dy;
                        int idx = y * Width + x;

                        Color32 c = pixels[idx];

                        // Convert to opponent color space (approximation to CIELab)
                        float r = c.r / 255f;
                        float g = c.g / 255f;
                        float b = c.b / 255f;

                        float l = 0.2126f * r + 0.7152f * g + 0.0722f * b; // Luminance
                        float a = r - g;  // Red-Green opponent
                        float bVal = 0.5f * (r + g) - b;  // Blue-Yellow opponent

                        sumL += l;
                        sumA += a;
                        sumB += bVal;
                        sumLL += l * l;
                        sumAA += a * a;
                        sumBB += bVal * bVal;
                        sumLA += l * a;
                        sumLB += l * bVal;
                        sumAB += a * bVal;
                        n++;
                    }
                }

                // Compute covariance matrix elements
                float meanL = sumL / n;
                float meanA = sumA / n;
                float meanB = sumB / n;

                float varL = sumLL / n - meanL * meanL;
                float varA = sumAA / n - meanA * meanA;
                float varB = sumBB / n - meanB * meanB;
                float covLA = sumLA / n - meanL * meanA;
                float covLB = sumLB / n - meanL * meanB;
                float covAB = sumAB / n - meanA * meanB;

                // Determinant of 3x3 covariance matrix
                // det = varL*(varA*varB - covAB^2) - covLA*(covLA*varB - covAB*covLB) + covLB*(covLA*covAB - varA*covLB)
                float det = varL * (varA * varB - covAB * covAB)
                          - covLA * (covLA * varB - covAB * covLB)
                          + covLB * (covLA * covAB - varA * covLB);

                // Volume = cube root of determinant (or sqrt for 2D approximation as in paper)
                // We use the absolute value to handle numerical issues
                float volume = Mathf.Pow(Mathf.Abs(det), 1f / 3f);

                return volume;
            }
            
            // Contrast congestion: local RMS contrast variability
            private float ComputeLocalContrastCongestion(NativeArray<Color32> pixels, int cx, int cy) {
                float sumContrast = 0f;
                float sumContrastSq = 0f;
                int n = 0;

                // Center luminance for reference
                int centerIdx = cy * Width + cx;
                Color32 centerC = pixels[centerIdx];
                float centerL = 0.2126f * (centerC.r / 255f) + 0.7152f * (centerC.g / 255f) + 0.0722f * (centerC.b / 255f);

                for (int dy = -WindowRadius; dy <= WindowRadius; dy++) {
                    for (int dx = -WindowRadius; dx <= WindowRadius; dx++) {
                        if (dx == 0 && dy == 0) continue;

                        int x = cx + dx;
                        int y = cy + dy;
                        int idx = y * Width + x;

                        Color32 c = pixels[idx];
                        float l = 0.2126f * (c.r / 255f) + 0.7152f * (c.g / 255f) + 0.0722f * (c.b / 255f);

                        // Weber contrast
                        float contrast = Mathf.Abs(l - centerL) / Mathf.Max(centerL, 0.01f);
                        sumContrast += contrast;
                        sumContrastSq += contrast * contrast;
                        n++;
                    }
                }

                // Standard deviation of local contrast values (congestion = variability)
                float meanContrast = sumContrast / n;
                float varContrast = sumContrastSq / n - meanContrast * meanContrast;

                return Mathf.Sqrt(Mathf.Max(varContrast, 0f));
            }

            // Orientation congestion: variability of local edge orientations
            // Uses simple Sobel-based gradient computation
            private float ComputeLocalOrientationCongestion(NativeArray<Color32> pixels, int cx, int cy) {
                // Collect gradient orientations in local window
                float sumSin = 0f;
                float sumCos = 0f;
                float sumMagnitude = 0f;
                int n = 0;

                for (int dy = -WindowRadius + 1; dy <= WindowRadius - 1; dy++) {
                    for (int dx = -WindowRadius + 1; dx <= WindowRadius - 1; dx++) {
                        int x = cx + dx;
                        int y = cy + dy;

                        // Compute gradient using Sobel operator
                        float gx = ((Color)pixels[y * Width + x + 1]).grayscale -
                                   ((Color)pixels[y * Width + x - 1]).grayscale;

                        float gy = ((Color)pixels[(y + 1) * Width + x]).grayscale -
                                   ((Color)pixels[(y - 1) * Width + x]).grayscale;

                        float magnitude = Mathf.Sqrt(gx * gx + gy * gy);

                        if (magnitude > 0.01f) {
                            // Double the angle (standard technique for orientation coherence)
                            float angle = Mathf.Atan2(gy, gx) * 2f;
                            sumSin += magnitude * Mathf.Sin(angle);
                            sumCos += magnitude * Mathf.Cos(angle);
                            sumMagnitude += magnitude;
                            n++;
                        }
                    }
                }

                if (n == 0 || sumMagnitude < 0.01f) return 0f;

                // Orientation coherence (1 = all same orientation, 0 = random)
                float coherence = Mathf.Sqrt(sumSin * sumSin + sumCos * sumCos) / sumMagnitude;

                // Congestion = 1 - coherence (high congestion when orientations are random)
                return 1f - coherence;
            }
        }
        
        public static void ResetValues() {
            ClutterReductionResults.Clear();
        }

        public static double GetAverageClutterReduction() {
            return ClutterReductionResults.Count > 0 ? ClutterReductionResults.Average() : 0;
        }
    }
}