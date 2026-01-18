using System.Collections.Generic;
using InpaintAR.Scripts.Util;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // FMM Inpainting (Fast Marching Method)
    // See DOI 10.1080/10867651.2004.10487596
    public class FastMarchingAlgorithm : AbstractInpaintingAlgorithm {
        // Use bytes for Pixel State Decision, as its very memory efficient for the quest
        private const byte Known = 0; // From Source (outside mask)
        private const byte Band = 1; // On Boundary
        private const byte Inside = 2; // Inside Inpainting Region
        
        private static readonly int[] Dir4X = { -1, 1, 0, 0 };
        private static readonly int[] Dir4Y = { 0, 0, -1, 1 };

        // Radius for Neighborhood of Inpainting Reference Pixels
        private const int Epsilon = 3;

        // min grad for inpainting
        private const float MinGradMagnitude = 1e-5f;
        private const float MinWeightVal = 0.001f;

        private int m_width;
        private int m_height;
        private byte[] m_flags;
        private float[] m_distances;
        private Color[] m_pixels;

        // Precomputed smoothed T field and its gradient (otherwise unstable)
        private float[] m_smoothedT;
        private float[] m_gradTx;
        private float[] m_gradTy;

        // Priority queue for the narrow band (min-heap based on distance)
        private PriorityQueue<int, float> m_narrowBand;

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            m_width = TextureUtility.GetImageWidth(source);
            m_height = TextureUtility.GetImageHeight(source);
            int pixelCount = m_width * m_height;

            m_sourcePixelBuffer = source.GetPixels32();

            if (m_inpaintedPixelBuffer == null || m_inpaintedPixelBuffer.Length != pixelCount) {
                m_inpaintedPixelBuffer = new Color32[pixelCount];
            }

            System.Array.Copy(TextureUtility.GetEmptyImagePixels(source), m_inpaintedPixelBuffer, pixelCount);

            InpaintFmm(maskPixelIndices);

            Texture2D resultImage = new Texture2D(m_width, m_height, TextureFormat.RGBA32, false);
            resultImage.SetPixels32(m_inpaintedPixelBuffer);
            resultImage.Apply();

            return resultImage;
        }

        private void InpaintFmm(HashSet<int> maskPixelIndices) {
            int pixelCount = m_width * m_height;

            m_flags = new byte[pixelCount];
            m_distances = new float[pixelCount];

            m_pixels = new Color[pixelCount];
            for (int i = 0; i < pixelCount; i++) {
                m_pixels[i] = m_sourcePixelBuffer[i];
            }

            m_narrowBand = new PriorityQueue<int, float>();

            // init vals 0 in both cases is known constant and distance 0, mask pixels need different init val
            foreach (int i in maskPixelIndices) {
                m_flags[i] = Inside;
                m_distances[i] = float.PositiveInfinity;
            }

            // Boundary Normal should not be calculated on the fly due to instability
            PrecomputeDistanceFieldAndGradient(maskPixelIndices);

            InitializeNarrowBand(maskPixelIndices);

            // Boundary not empty
            while (m_narrowBand.Count > 0) {
                int i = m_narrowBand.Dequeue();

                (int col, int row) = ToCoords(i);

                // f(i,j) = KNOWN
                m_flags[i] = Known;

                // for (k,l) in (i-1,j),(i,j-1),(i+1,j),(i,j+1)
                for (int n = 0; n < 4; n++) {
                    int curCol = col + Dir4X[n];
                    int curRow = row + Dir4Y[n];

                    if (!IsInBounds(curCol, curRow)) continue;

                    int curI = ToIndex(curCol, curRow);

                    if (m_flags[curI] == Known) continue;

                    // if pixel is not known
                    if (m_flags[curI] == Inside) {
                        m_flags[curI] = Band; // since inpainted -> new Boundary
                        InpaintPixel(curI);
                    }

                    m_distances[curI] = ComputeMinEikonalSolution(curCol, curRow, m_distances, m_flags);

                    // insert for next iterations
                    m_narrowBand.Enqueue(curI, m_distances[curI]);
                }
            }
        }

        private void PrecomputeDistanceFieldAndGradient(HashSet<int> maskPixelIndices) {
            int pixelCount = m_width * m_height;

            float[] tOut = RunFmm(maskPixelIndices, isOutward: true);
            float[] tIn = RunFmm(maskPixelIndices, isOutward: false);

            float[] combinedT = new float[pixelCount];
            for (int i = 0; i < pixelCount; i++) {
                combinedT[i] = maskPixelIndices.Contains(i) ? tIn[i] : -Mathf.Min(tOut[i], Epsilon);
            }

            m_smoothedT = new float[pixelCount];
            ApplyTentFilter(combinedT, m_smoothedT);

            m_gradTx = new float[pixelCount];
            m_gradTy = new float[pixelCount];
            ComputeGradientField(m_smoothedT, m_gradTx, m_gradTy);
        }

        private float[] RunFmm(HashSet<int> maskPixelIndices, bool isOutward) {
            int pixelCount = m_width * m_height;
            float[] t = new float[pixelCount];
            byte[] flags = new byte[pixelCount];

            InitializeFmmArrays(t, flags, maskPixelIndices, isOutward);

            var narrowBand = new PriorityQueue<int, float>();
            InitializeFmmBoundary(narrowBand, t, flags, maskPixelIndices, isOutward);

            PropagateFmm(narrowBand, t, flags, maskPixelIndices, isOutward);

            return t;
        }

        private static void InitializeFmmArrays(float[] t, byte[] flags, HashSet<int> maskPixelIndices, bool isOutward) {
            for (int i = 0; i < t.Length; i++) {
                if (isOutward) {
                    flags[i] = Inside;
                    t[i] = maskPixelIndices.Contains(i) ? 0 : float.PositiveInfinity;
                }
                else {
                    flags[i] = maskPixelIndices.Contains(i) ? Inside : Known;
                    t[i] = maskPixelIndices.Contains(i) ? float.PositiveInfinity : 0f;
                }
            }
        }

        private void InitializeFmmBoundary(PriorityQueue<int, float> narrowBand, float[] t, byte[] flags,
            HashSet<int> maskPixelIndices, bool isOutward) {
            int[] dx = Dir4X;
            int[] dy = Dir4Y;

            foreach (int i in maskPixelIndices) {
                (int x, int y) = ToCoords(i);

                for (int j = 0; j < 4; j++) {
                    int nx = x + dx[j];
                    int ny = y + dy[j];

                    if (!IsInBounds(nx, ny)) continue;

                    int nI = ToIndex(nx, ny);
                    if (maskPixelIndices.Contains(nI) == isOutward) continue;
                    if (flags[isOutward ? nI : i] == Band) continue;

                    int bandI = isOutward ? nI : i;
                    flags[bandI] = Band;
                    t[bandI] = 1f;
                    narrowBand.Enqueue(bandI, 1f);

                    if (!isOutward) break;
                }
            }
        }

        private void PropagateFmm(PriorityQueue<int, float> narrowBand, float[] t, byte[] flags,
            HashSet<int> maskPixelIndices, bool isOutward) {
            while (narrowBand.Count > 0) {
                int i = narrowBand.Dequeue();
                if (isOutward && t[i] > Epsilon) continue;
                flags[i] = Known;

                (int col, int row) = ToCoords(i);
                
                for (int n = 0; n < 4; n++) {
                    int curCol = col + Dir4X[n];
                    int curRow = row + Dir4Y[n];

                    if (!IsInBounds(curCol, curRow)) continue;

                    int curI = ToIndex(curCol, curRow);
                    if (flags[curI] == Known) continue;
                    if (maskPixelIndices.Contains(curI) == isOutward) continue;

                    float newT = ComputeMinEikonalSolution(curCol, curRow, t, flags);
                    if (newT >= t[curI]) continue;

                    t[curI] = newT;
                    flags[curI] = Band;
                    narrowBand.Enqueue(curI, newT);
                }
            }
        }

        private float ComputeMinEikonalSolution(int col, int row, float[] t, byte[] flags) {
            float t1 = SolveEikonal(col - 1, row, col, row - 1, t, flags);
            float t2 = SolveEikonal(col + 1, row, col, row - 1, t, flags);
            float t3 = SolveEikonal(col - 1, row, col, row + 1, t, flags);
            float t4 = SolveEikonal(col + 1, row, col, row + 1, t, flags);
            return Mathf.Min(Mathf.Min(t1, t2), Mathf.Min(t3, t4));
        }


        // 3x3 Tent Filter
        // 1 2 1
        // 2 4 2
        // 1 2 1
        private void ApplyTentFilter(float[] input, float[] output) {
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    int i = ToIndex(x, y);
                    output[i] = ComputeTentFilteredValue(input, x, y);
                }
            }
        }

        private float ComputeTentFilteredValue(float[] input, int x, int y) {
            float sum = 0f;
            float weightSum = 0f;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (!TryGetPixelValue(input, x + dx, y + dy, out float value)) continue;

                    float weight = (2 - Mathf.Abs(dx)) * (2 - Mathf.Abs(dy));
                    sum += value * weight;
                    weightSum += weight;
                }
            }

            return sum / weightSum;
        }

        private bool TryGetPixelValue(float[] array, int x, int y, out float value) {
            value = 0f;
            if (!IsInBounds(x, y)) return false;

            value = array[ToIndex(x, y)];
            return true;
        }


        private void ComputeGradientField(float[] t, float[] gradX, float[] gradY) {
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    int i = ToIndex(x, y);

                    gradX[i] = x switch {
                        > 0 when x < m_width - 1 => (t[i + 1] - t[i - 1]) * 0.5f,
                        > 0 => t[i] - t[i - 1],
                        _ => t[i + 1] - t[i]
                    };

                    gradY[i] = y switch {
                        > 0 when y < m_height - 1
                            => (t[i + m_width] - t[i - m_width]) * 0.5f,
                        > 0
                            => t[i] - t[i - m_width],
                        _
                            => t[i + m_width] - t[i]
                    };
                }
            }
        }

        private void InitializeNarrowBand(HashSet<int> maskPixelIndices) {
            foreach (int i in maskPixelIndices) {
                (int x, int y) = ToCoords(i);

                bool isBoundary = false;
                float minDist = float.PositiveInfinity;

                int[] dx = Dir4X;
                int[] dy = Dir4Y;

                for (int j = 0; j < 4; j++) {
                    int nx = x + dx[j];
                    int ny = y + dy[j];

                    // Bounds
                    if (!IsInBounds(nx, ny)) continue;
                    // Neighbor != known, no boundary -> continue
                    if (m_flags[ToIndex(nx, ny)] != Known) continue;

                    isBoundary = true;
                    // Distance to known pixel is 1
                    minDist = 1f;
                }

                if (!isBoundary) continue;

                m_flags[i] = Band;
                m_distances[i] = minDist;
                m_narrowBand.Enqueue(i, minDist);
            }
        }

        private float SolveEikonal(int i1, int j1, int i2, int j2, float[] t, byte[] flags) {
            bool valid1 = IsInBounds(i1, j1);
            bool valid2 = IsInBounds(i2, j2);

            int index1 = valid1 ? ToIndex(i1, j1) : -1;
            int index2 = valid2 ? ToIndex(i2, j2) : -1;

            bool known1 = valid1 && flags[index1] == Known;
            bool known2 = valid2 && flags[index2] == Known;

            switch (known1) {
                case true when known2: {
                    float t1 = t[index1];
                    float t2 = t[index2];
                    float diff = t1 - t2;
                    float disc = 2f - diff * diff;
                    if (disc <= 0f) return Mathf.Min(t1, t2) + 1f;
                    
                    float r = Mathf.Sqrt(disc);
                    float s = (t1 + t2 - r) * 0.5f;
                    if (s >= t1 && s >= t2) return s;
                    s += r;
                    if (s >= t1 && s >= t2) return s;

                    return Mathf.Min(t1, t2) + 1f;
                }
                case true:
                    return t[index1] + 1f;
            }

            if (known2) return t[index2] + 1f;

            return float.PositiveInfinity;
        }

        private void InpaintPixel(int i) {
            (int x, int y) = ToCoords(i);

            // Vector3 used, as it does not need Heap Allocation -> more performant/efficient
            // Also a lot cleaner than using individual variables 
            Vector3 sumRGB = Vector3.zero;
            float sumA = 0f;
            float totalWeight = 0f;

            Vector2 normal = GetNormalizedGradient(i);
            int eps = Epsilon;

            for (int dy = -eps; dy <= eps; dy++) {
                for (int dx = -eps; dx <= eps; dx++) {
                    if (dx == 0 && dy == 0) continue;

                    int nx = x + dx;
                    int ny = y + dy;

                    if (!IsValidKnownNeighbor(nx, ny, dx, dy, out int nI)) continue;

                    float weight = ComputeWeight(dx, dy, i, nI, normal);
                    AccumulateColor(nI, dx, dy, weight, ref sumRGB, ref sumA);
                    totalWeight += weight;
                }
            }

            if (totalWeight <= 0f) return;

            ApplyInpaintedColor(i, sumRGB, sumA, totalWeight);
        }


        private Vector2 GetNormalizedGradient(int i) {
            float gradTx = m_gradTx[i];
            float gradTy = m_gradTy[i];
            float magnitude = Mathf.Sqrt(gradTx * gradTx + gradTy * gradTy);

            return magnitude > MinGradMagnitude ? new Vector2(gradTx / magnitude, gradTy / magnitude) : new Vector2(gradTx, gradTy);
        }

        private bool IsValidKnownNeighbor(int nx, int ny, int dx, int dy, out int nI) {
            nI = -1;
            if (!IsInBounds(nx, ny)) return false;

            nI = ToIndex(nx, ny);
            if (m_flags[nI] != Known) return false;

            float dist = Mathf.Sqrt(dx * dx + dy * dy);
            return dist <= Epsilon;
        }

        private float ComputeWeight(int dx, int dy, int i, int nI, Vector2 normal) {
            float dist = Mathf.Sqrt(dx * dx + dy * dy);
            float dirFactor = Mathf.Max((-dx * normal.x + -dy * normal.y) / dist, MinWeightVal);
            float dstFactor = 1f / (dist * dist);
            float levFactor = 1f / (1f + Mathf.Abs(m_distances[nI] - m_distances[i]));
            return dirFactor * dstFactor * levFactor;
        }

        private void AccumulateColor(int i, int dx, int dy, float weight,
            ref Vector3 sumRGB, ref float sumA) {
            (int x, int y) = ToCoords(i);
            Color neighborColor = m_pixels[i];

            ComputeGradientI(x, y, out var gradIx, out var gradIy);

            sumRGB += weight * new Vector3(
                neighborColor.r + gradIx.x * (-dx) + gradIy.x * (-dy),
                neighborColor.g + gradIx.y * (-dx) + gradIy.y * (-dy),
                neighborColor.b + gradIx.z * (-dx) + gradIy.z * (-dy)
            );
            sumA += weight * neighborColor.a;
        }


        private void ApplyInpaintedColor(int i, Vector3 sumRGB, float sumA, float totalWeight) {
            Color inpaintedColor = new Color(
                Mathf.Clamp01(sumRGB.x / totalWeight),
                Mathf.Clamp01(sumRGB.y / totalWeight),
                Mathf.Clamp01(sumRGB.z / totalWeight),
                Mathf.Clamp01(sumA / totalWeight)
            );

            m_pixels[i] = inpaintedColor;
            m_inpaintedPixelBuffer[i] = inpaintedColor;
        }

        private void ComputeGradientI(int x, int y, out Vector3 gradX, out Vector3 gradY) {
            int i = ToIndex(x, y);
            Color center = m_pixels[i];

            gradX = ComputeGradient1D(x, y, 1, 0, center);
            gradY = ComputeGradient1D(x, y, 0, 1, center);
        }

        private Vector3 ComputeGradient1D(int x, int y, int dx, int dy, Color center) {
            int prevX = x - dx, prevY = y - dy;
            int nextX = x + dx, nextY = y + dy;

            bool hasPrev = IsInBounds(prevX, prevY);
            bool hasNext = IsInBounds(nextX, nextY);

            int prevI = hasPrev ? ToIndex(prevX, prevY) : -1;
            int nextI = hasNext ? ToIndex(nextX, nextY) : -1;

            bool knownPrev = hasPrev && m_flags[prevI] == Known;
            bool knownNext = hasNext && m_flags[nextI] == Known;

            Color next;
            switch (knownPrev) {
                case true when knownNext: {
                    Color prev = m_pixels[prevI];
                    next = m_pixels[nextI];
                    return new Vector3((next.r - prev.r) * 0.5f, (next.g - prev.g) * 0.5f, (next.b - prev.b) * 0.5f);
                }
                case true: {
                    Color prev = m_pixels[prevI];
                    return new Vector3(center.r - prev.r, center.g - prev.g, center.b - prev.b);
                }
            }

            if (!knownNext) return Vector3.zero;
            next = m_pixels[nextI];
            return new Vector3(next.r - center.r, next.g - center.g, next.b - center.b);
        }
        
        private bool IsInBounds(int col, int row) => col >= 0 && col < m_width && row >= 0 && row < m_height;
        
        private int ToIndex(int col, int row) => row * m_width + col;
        
        private (int col, int row) ToCoords(int idx) => (idx % m_width, idx / m_width);
    }
}