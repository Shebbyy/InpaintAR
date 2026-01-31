using System.Collections.Generic;
using System.Linq;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

namespace InpaintAR.Scripts.Benchmarking.Evaluators {
    // Image Inpainting Quality Assessment (IIQA) based on Guided Regional Statistics (GRS)
    // See DOI 10.1587/transinf.2018EDL8206
    public class QualityEvaluator {
        private const float StructuralSimilarityWeight = 0.6f;
        private static readonly List<float> QualityResults = new();
        private static JobHandle? _jobHandle;
        private static NativeArray<Color32> _nativeOriginal;
        private static NativeArray<Color32> _nativeInpainted;
        private static NativeArray<float> _structuralSimilarity;
        private static NativeArray<float> _naturalness;
        private static NativeArray<byte> _nativeMask;

        public static void EvaluateQuality(Color32[] originalImage, Color32[] inpaintedImage, int width, int height,
            HashSet<int> maskPixelIndices) {
            // only one evaluation can run at a time
            if (_jobHandle is not null) {
                if (!_jobHandle.Value.IsCompleted) return;
                
                float ssim = _structuralSimilarity[0];
                float nat = _naturalness[0];
                // Q = α · f1 + (1 − α) · f2
                float overallQuality = StructuralSimilarityWeight * ssim + (1 - StructuralSimilarityWeight) * nat;

                _nativeOriginal.Dispose();
                _nativeInpainted.Dispose();
                _nativeMask.Dispose();
                _structuralSimilarity.Dispose();
                _naturalness.Dispose();

                QualityResults.Add(overallQuality);
            }

            _nativeOriginal  = new NativeArray<Color32>(originalImage, Allocator.TempJob);
            _nativeInpainted = new NativeArray<Color32>(inpaintedImage, Allocator.TempJob);

            // Needs to be NativeArray to allow for Burst Compiled Job
            _nativeMask = new NativeArray<byte>(width * height, Allocator.TempJob);
            for (int i = 0; i < width * height; i++) {
                _nativeMask[i] = 0;
            }
            // more efficient than .Contains inside of the upper for loop due so little pixels being part of the mask
            foreach (int index in maskPixelIndices) {
                _nativeMask[index] = 1;
            }

            _structuralSimilarity = new NativeArray<float>(1, Allocator.TempJob);
            _naturalness          = new NativeArray<float>(1, Allocator.TempJob);

            var qualityJob = new QualityEvaluationJob {
                OriginalPixels = _nativeOriginal,
                InpaintedPixels = _nativeInpainted,
                Mask = _nativeMask,
                Width = width,
                Height = height,
                StructuralSimilarity = _structuralSimilarity,
                Naturalness = _naturalness
            };

            _jobHandle = qualityJob.Schedule();
        }

        public static void ResetValues() {
            QualityResults.Clear();
        }

        public static double GetAverageQuality() {
            return QualityResults.Count > 0 ? QualityResults.Average() : -1;
        }

        [BurstCompile]
        private struct QualityEvaluationJob : IJob {
            [ReadOnly] public NativeArray<Color32> OriginalPixels;
            [ReadOnly] public NativeArray<Color32> InpaintedPixels;
            [ReadOnly] public NativeArray<byte> Mask;
            public int Width;
            public int Height;

            [WriteOnly] public NativeArray<float> StructuralSimilarity;
            [WriteOnly] public NativeArray<float> Naturalness;

            public void Execute() {
                float structuralSimilarity = ComputeStructuralSimilarity();

                float naturalness = ComputeNaturalness();

                StructuralSimilarity[0] = structuralSimilarity;
                Naturalness[0] = naturalness;
            }

            private float ComputeStructuralSimilarity() {
                const int numBins = 256; // bits, 32 byte color -> 256 values
                NativeArray<int> histInterior = new NativeArray<int>(numBins, Allocator.Temp);
                NativeArray<int> histExterior = new NativeArray<int>(numBins, Allocator.Temp);
    
                int interiorCount = 0;
                int exteriorCount = 0;

                for (int y = 1; y < Height - 1; y++) {
                    for (int x = 1; x < Width - 1; x++) {
                        int idx = y * Width + x;
                        

                        if (Mask[idx] == 1) {
                            float gradient = ComputeGradientMagnitude(x, y, InpaintedPixels);
                            int binIndex = Mathf.Clamp((int)(gradient * (numBins - 1)), 0, numBins - 1);
                            histInterior[binIndex]++;
                            interiorCount++;
                        } else {
                            float gradient = ComputeGradientMagnitude(x, y, InpaintedPixels);
                            int binIndex = Mathf.Clamp((int)(gradient * (numBins - 1)), 0, numBins - 1);
                            histExterior[binIndex]++;
                            exteriorCount++;
                        }
                    }
                }

                // Probability Distribution exterior (original) / interior (inpainted)
                NativeArray<float> pint = new NativeArray<float>(numBins, Allocator.Temp);
                NativeArray<float> pext = new NativeArray<float>(numBins, Allocator.Temp);

                for (int i = 0; i < numBins; i++) {
                    pint[i] = interiorCount > 0 ? (float)histInterior[i] / interiorCount : 0f;
                    pext[i] = exteriorCount > 0 ? (float)histExterior[i] / exteriorCount : 0f;
                }

                // Get distance of the two distributions
                float kldIntExt = ComputeKld(pint, pext, numBins);
                float kldExtINT = ComputeKld(pext, pint, numBins);
                float f1 = 0.5f * (kldIntExt + kldExtINT);

                histInterior.Dispose();
                histExterior.Dispose();
                pint.Dispose();
                pext.Dispose();

                // Normalize to [0, 1] range (lower KLD = better quality)
                return 1f / (1f + f1);
            }

            private static float ComputeKld(NativeArray<float> par1, NativeArray<float> par2, int numBins) {
                float kld = 0f;
                const float epsilon = 1e-10f; 

                for (int i = 0; i < numBins; i++) {
                    if (par1[i] > epsilon) {
                        float q = Mathf.Max(par2[i], epsilon); // Avoid log(0)
                        kld += par1[i] * Mathf.Log(par1[i] / q);
                    }
                }

                return kld;
            }
            
            private float ComputeNaturalness() {
                float nfOriginal = ComputeNaturalnessFeature(OriginalPixels);
                float nfInpainted = ComputeNaturalnessFeature(InpaintedPixels);

                // f2 = |NF_Original - NF_Inpainted|
                float f2 = Mathf.Abs(nfOriginal - nfInpainted);

                // Normalize to [0, 1] range (lower difference = better quality)
                return 1f / (1f + f2);
            }
            
            private float ComputeNaturalnessFeature(NativeArray<Color32> pixels) {
                NativeList<float> gradients = new NativeList<float>(Allocator.Temp);

                for (int y = 1; y < Height - 1; y++) {
                    for (int x = 1; x < Width - 1; x++) {
                        int idx = y * Width + x;
                        if (Mask[idx] == 1 && HasNonMaskNeighbor(x, y)) {
                            float gradient = ComputeGradientMagnitude(x, y, pixels);
                            gradients.Add(gradient);
                        }
                    }
                }

                if (gradients.Length == 0) return 0f;

                // Compute T1: median
                float t1 = ComputeT1Parameter(gradients);

                // Compute T2: mean absolute deviation
                float t2 = ComputeT2Parameter(gradients);

                gradients.Dispose();

                // Prior Values taken from Paper
                const float t1Prior = 0.38f;
                const float t2Prior = 0.14f;
                const float theta = 0.5f;

                // NF = (1-Theta) * T1/T1_pr + Theta * T2/T2_pr
                float nf = (1f - theta) * (t1 / t1Prior) + theta * (t2 / t2Prior);

                return nf;
            }
            
            private static float ComputeT1Parameter(NativeList<float> gradients) {
                if (gradients.Length == 0) return 0f;

                NativeArray<float> sorted = new NativeArray<float>(gradients.Length, Allocator.Temp);
                gradients.AsArray().CopyTo(sorted);
                sorted.Sort();

                float median = sorted[sorted.Length / 2];
                sorted.Dispose();

                return median;
            }
            
            private static float ComputeT2Parameter(NativeList<float> gradients) {
                if (gradients.Length == 0) return 0f;

                float mean = 0f;
                for (int i = 0; i < gradients.Length; i++) {
                    mean += gradients[i];
                }
                mean /= gradients.Length;

                float mad = 0f;
                for (int i = 0; i < gradients.Length; i++) {
                    mad += Mathf.Abs(gradients[i] - mean);
                }
                mad /= gradients.Length;

                return mad;
            }

            private bool HasNonMaskNeighbor(int x, int y) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;

                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0
                            && nx < Width
                            && ny >= 0
                            && ny < Height) {
                            int nidx = ny * Width + nx;
                            if (Mask[nidx] == 0) return true;
                        }
                    }
                }

                return false;
            }

            private float ComputeGradientMagnitude(int x, int y, NativeArray<Color32> pixels) {
                int idx = y * Width + x;
                float gradX = 0f;
                float gradY = 0f;

                if (x > 0 && x < Width - 1) {
                    gradX = ((Color)pixels[idx + 1]).grayscale - ((Color)pixels[idx - 1]).grayscale;
                }

                if (y > 0 && y < Height - 1) {
                    gradY = ((Color)pixels[idx + Width]).grayscale - ((Color)pixels[idx - Width]).grayscale;
                }

                return Mathf.Sqrt(gradX * gradX + gradY * gradY);
            }
        }
    }
}