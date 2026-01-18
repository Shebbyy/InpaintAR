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
        private const int Epsilon = 2;

        // min grad for inpainting
        private const float MinGradMagnitude = 1e-5f;
        private const float MinWeightVal = 0.001f;

        private int m_width;
        private int m_height;
        private int m_pixelCount;
        private byte[] m_flags;
        private float[] m_distances;

        // Precomputed smoothed T field and its gradient (otherwise unstable)
        private float[] m_smoothedT;
        private float[] m_gradTx;
        private float[] m_gradTy;
        private bool[] m_mask;

        // Priority queue for the Boundary (min-heap based on distance)
        private PriorityQueue<int, float> m_boundary;

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            m_width = TextureUtility.GetImageWidth(source);
            m_height = TextureUtility.GetImageHeight(source);
            m_pixelCount = m_width * m_height;

            if (MInpaintedPixelBuffer == null || MInpaintedPixelBuffer.Length != m_pixelCount) {
                MInpaintedPixelBuffer = new Color32[m_pixelCount];
            }

            m_mask = new bool[m_pixelCount];
            foreach (var hashVal in maskPixelIndices) {
                m_mask[hashVal] = true;
            }

            Texture2D resultImage = new Texture2D(m_width, m_height, TextureFormat.RGBA32, false);

            MSourcePixelBuffer = source.GetPixels32();

            System.Array.Copy(MSourcePixelBuffer, MInpaintedPixelBuffer, m_pixelCount);

            InpaintFmm(maskPixelIndices);

            resultImage.SetPixels32(MInpaintedPixelBuffer);
            resultImage.Apply();

            return resultImage;
        }

        private void InpaintFmm(HashSet<int> maskPixelIndices) {
            m_flags = new byte[m_pixelCount];
            m_distances = new float[m_pixelCount];

            m_boundary = new PriorityQueue<int, float>();

            // init vals 0 in both cases is known constant and distance 0, mask pixels need different init val
            foreach (int i in maskPixelIndices) {
                m_flags[i] = Inside;
                m_distances[i] = float.PositiveInfinity;
            }

            // Boundary Normal should not be calculated on the fly due to instability
            PrecomputeDistanceFieldAndGradient(maskPixelIndices);

            InitializeNarrowBand(maskPixelIndices);

            // Boundary not empty
            int inpaintedCount = 0;
            while (m_boundary.Count > 0) {
                int i = m_boundary.Dequeue();

                if (m_flags[i] == Known) continue;

                (int col, int row) = ToCoords(i);

                // f(i,j) = KNOWN
                m_flags[i] = Known;

                // for (k,l) in (i-1,j),(i,j-1),(i+1,j),(i,j+1)
                for (int n = 0; n < 4; n++) {
                    int curCol = col + Dir4X[n];
                    int curRow = row + Dir4Y[n];

                    if (!IsInBounds(curCol, curRow)) continue;

                    int currentIndex = ToIndex(curCol, curRow);

                    switch (m_flags[currentIndex]) {
                        case Known:
                            continue;
                        // if pixel is not known
                        case Inside:
                            m_flags[currentIndex] = Band; // since inpainted -> new Boundary
                            InpaintPixel(currentIndex);
                            inpaintedCount++;
                            break;
                    }

                    m_distances[currentIndex] = ComputeMinEikonalSolution(curCol, curRow, m_distances, m_flags);

                    // insert for next iterations
                    m_boundary.Enqueue(currentIndex, m_distances[currentIndex]);
                }
            }
            Debug.Log($"Inpainted {inpaintedCount} pixels, mask size: {maskPixelIndices.Count}");
        }

        private void PrecomputeDistanceFieldAndGradient(HashSet<int> maskPixelIndices) {
            float[] tOut = RunFmm(maskPixelIndices, isOutward: true);
            float[] tIn = RunFmm(maskPixelIndices, isOutward: false);

            float[] combinedT = new float[m_pixelCount];
            for (int i = 0; i < m_pixelCount; i++) {
                combinedT[i] = m_mask[i] ? tIn[i] : -Mathf.Min(tOut[i], Epsilon);
            }

            m_smoothedT = new float[m_pixelCount];
            ApplyTentFilter(combinedT, m_smoothedT);

            m_gradTx = new float[m_pixelCount];
            m_gradTy = new float[m_pixelCount];
            ComputeGradientField(m_smoothedT, m_gradTx, m_gradTy);
        }

        private float[] RunFmm(HashSet<int> maskPixelIndices, bool isOutward) {
            float[] t = new float[m_pixelCount];
            byte[] flags = new byte[m_pixelCount];

            InitializeFmmArrays(t, flags, isOutward);

            var narrowBand = new PriorityQueue<int, float>();
            InitializeFmmBoundary(narrowBand, t, flags, maskPixelIndices, isOutward);

            PropagateFmm(narrowBand, t, flags, isOutward);

            return t;
        }

        private void InitializeFmmArrays(float[] t, byte[] flags, bool isOutward) {
            for (int i = 0; i < t.Length; i++) {
                if (isOutward) {
                    // Outward: propagate FROM mask TO outside
                    flags[i] = m_mask[i] ? Known : Inside;
                    t[i] = m_mask[i] ? 0f : float.PositiveInfinity;
                }
                else {
                    // Inward: propagate FROM outside TO mask
                    flags[i] = m_mask[i] ? Inside : Known;
                    t[i] = m_mask[i] ? float.PositiveInfinity : 0f;
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

                    int neighborIndex = ToIndex(nx, ny);

                    if (m_mask[neighborIndex]) continue;
                    if (isOutward) {
                        if (flags[neighborIndex] == Band) continue;

                        flags[neighborIndex] = Band;
                        t[neighborIndex] = 1f;
                        narrowBand.Enqueue(neighborIndex, 1f);
                    }
                    else {
                        if (flags[i] == Band) continue;

                        flags[i] = Band;
                        t[i] = 1f;
                        narrowBand.Enqueue(i, 1f);
                        break; // Only need to mark this mask pixel once
                    }
                }
            }
        }

        private void PropagateFmm(PriorityQueue<int, float> narrowBand, float[] t, byte[] flags
            , bool isOutward) {
            while (narrowBand.Count > 0) {
                int i = narrowBand.Dequeue();

                if (flags[i] == Known) continue;

                if (isOutward && t[i] > Epsilon) continue;
                flags[i] = Known;

                (int col, int row) = ToCoords(i);

                for (int n = 0; n < 4; n++) {
                    int curCol = col + Dir4X[n];
                    int curRow = row + Dir4Y[n];

                    if (!IsInBounds(curCol, curRow)) continue;

                    int curI = ToIndex(curCol, curRow);
                    if (flags[curI] == Known) continue;
                    if (m_mask[curI] == isOutward) continue;

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
            const int maxWeight = 16;
            // Process interior pixels without bounds checks
            for (int y = 1; y < m_height - 1; y++) {
                for (int x = 1; x < m_width - 1; x++) {
                    int i = ToIndex(x, y);
                    // calculated directly here without bounds check -> more performance
                    output[i] = (
                        input[i - m_width - 1] + 2f * input[i - m_width] + input[i - m_width + 1] +
                        2f * input[i - 1] + 4f * input[i] + 2f * input[i + 1] +
                        input[i + m_width - 1] + 2f * input[i + m_width] + input[i + m_width + 1]
                    ) / maxWeight;
                }
            }

            // Handle edges separately (less frequent)
            for (int x = 0; x < m_width; x++) {
                output[ToIndex(x, 0)] = ComputeTentFilteredValue(input, x, 0);
                output[ToIndex(x, m_height - 1)] = ComputeTentFilteredValue(input, x, m_height - 1);
            }
            for (int y = 1; y < m_height - 1; y++) {
                output[ToIndex(0, y)] = ComputeTentFilteredValue(input, 0, y);
                output[ToIndex(m_width - 1, y)] = ComputeTentFilteredValue(input, m_width - 1, y);
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
                m_boundary.Enqueue(i, minDist);
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

            Debug.Log($"Inpaint Total Weight = {totalWeight}");
            if (totalWeight <= 0f) return;

            ApplyInpaintedColor(i, sumRGB, sumA, totalWeight);
        }


        private Vector2 GetNormalizedGradient(int i) {
            float gradTx = m_gradTx[i];
            float gradTy = m_gradTy[i];
            float magnitude = Mathf.Sqrt(gradTx * gradTx + gradTy * gradTy);

            return magnitude > MinGradMagnitude
                ? new Vector2(gradTx / magnitude, gradTy / magnitude)
                : new Vector2(gradTx, gradTy);
        }

        private bool IsValidKnownNeighbor(int nx, int ny, int dx, int dy, out int nI) {
            nI = -1;
            if (!IsInBounds(nx, ny)) return false;

            nI = ToIndex(nx, ny);
            if (m_flags[nI] != Known && m_flags[nI] != Band) return false;

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
            Color neighborColor = MInpaintedPixelBuffer[i];
        
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

            MInpaintedPixelBuffer[i] = inpaintedColor;
        }

        private void ComputeGradientI(int x, int y, out Vector3 gradX, out Vector3 gradY) {
            int i = ToIndex(x, y);
            Color center = MInpaintedPixelBuffer[i];

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
                    Color prev = MInpaintedPixelBuffer[prevI];
                    next = MInpaintedPixelBuffer[nextI];
                    return new Vector3((next.r - prev.r) * 0.5f, (next.g - prev.g) * 0.5f, (next.b - prev.b) * 0.5f);
                }
                case true: {
                    Color prev = MInpaintedPixelBuffer[prevI];
                    return new Vector3(center.r - prev.r, center.g - prev.g, center.b - prev.b);
                }
            }

            if (!knownNext) return Vector3.zero;
            next = MInpaintedPixelBuffer[nextI];
            return new Vector3(next.r - center.r, next.g - center.g, next.b - center.b);
        }

        private bool IsInBounds(int col, int row) => col >= 0 && col < m_width && row >= 0 && row < m_height;

        private int ToIndex(int col, int row) => row * m_width + col;

        private (int col, int row) ToCoords(int idx) => (idx % m_width, idx / m_width);
    }
}