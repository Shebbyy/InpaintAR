using System.Collections.Generic;
using InpaintAR.Scripts.Util;
using JetBrains.Annotations;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;

namespace InpaintAR.Scripts.SnakeEdgeDetection {
    public class BalloonSnake {
        // Profiler markers for key operations
        private readonly ProfilerMarker m_sEdgeMapMarker = new("SnakeEdge.ComputeEdgeMap");
        private readonly ProfilerMarker m_sEvolveSnakeMarker = new("SnakeEdge.EvolveSnake");
        
        // Might need fine tuning dependent on lighting, further analysis required if time
        private const float Elasticity = 1.5f; // increase for smoother contours
        private const float Rigidity = 1f; // increase to prevent breaking apart
        private const float PositionScaling = 0.12f; // How much the Position change gets scaled in total
        private const float MovementPerFrame = 5f; // movement per frame
        private const float EdgeAttraction = 150.0f; // increased edge attraction
        private const float EdgeThreshold = 0.025f; // Threshold for edge detection - lower to detect more edges
        private const float EdgeDampenFactor = 20.0f;
        
        private const float BarrierWeight = 125.0f; // Weight of the counter power when close
        private const float StabilizationThreshold = 0.015f; // Average movement threshold for early stopping

        private const float MinGradMagnitude = 1e-5f;

        private Vector2[,] m_gradientField;
        private float[,] m_edgeMap;
        private float[,] m_gray;
        private HashSet<int> m_cachedRefinedMask;
        private List<Vector2> m_cachedContourPoints;
        
        public HashSet<int> ApplyBalloonSnake(Texture2D fillImageTexture, Rect maskBounds, int iterations, int width, int height) {
            NativeArray<Color32> pixels = fillImageTexture.GetPixelData<Color32>(0);

            int margin = 20;
            Rect computeRegion = ExpandRect(maskBounds, margin, width, height);

            ComputeSobelEdgeMapInRegion(pixels, width, height, computeRegion);
            
            var snakePoints = EvolveBalloonSnake(m_cachedContourPoints, m_edgeMap, width, height, iterations);

            m_cachedContourPoints = snakePoints;

            TextureUtility.FillContour(m_cachedRefinedMask, snakePoints, width);

            return m_cachedRefinedMask;
        }
        
        [CanBeNull]
        public HashSet<int> GetRefinedMask() {
            return m_cachedRefinedMask;
        }

        [CanBeNull]
        public List<Vector2> GetContourPoints() {
            return m_cachedContourPoints;
        }

        public void ResetSelectionMask() {
            m_cachedRefinedMask = null;
            m_cachedContourPoints = null;
        }

        public void InitializeCacheVariables(int width, int height) {
            m_gradientField = new Vector2[width, height];
            m_edgeMap = new float[width, height];
            m_gray = new float[width, height];
            m_cachedRefinedMask = new HashSet<int>();
        }

        public void SetContourPoints(List<Vector2> newContourPoints) {
            m_cachedContourPoints = newContourPoints;
        }
        
        private static Rect ExpandRect(Rect rect, int margin, int width, int height) {
            return new Rect(
                Mathf.Max(0, rect.x - margin),
                Mathf.Max(0, rect.y - margin),
                Mathf.Min(width - rect.x + margin, rect.width + 2 * margin),
                Mathf.Min(height - rect.y + margin, rect.height + 2 * margin)
            );
        }

        private void ComputeSobelEdgeMapInRegion(NativeArray<Color32> pixels, int width, int height, Rect region) {
            m_sEdgeMapMarker.Begin();
            
            int minX = Mathf.Max(0, (int)region.x);
            int maxX = Mathf.Min(width, (int)(region.x + region.width));
            int minY = Mathf.Max(0, (int)region.y);
            int maxY = Mathf.Min(height, (int)(region.y + region.height));

            ComputeGrayscaleMask(pixels, width, minX, maxX, minY, maxY);

            float maxEdgeEnergy = 0f;

            // Sobel: compute squared gradient magnitude |∇I|²
            for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                    float gx =
                        -m_gray[x - 1, y - 1] - 2f * m_gray[x - 1, y] - m_gray[x - 1, y + 1] +
                        m_gray[x + 1, y - 1] + 2f * m_gray[x + 1, y] + m_gray[x + 1, y + 1];

                    float gy =
                        -m_gray[x - 1, y - 1] - 2f * m_gray[x, y - 1] - m_gray[x + 1, y - 1] +
                        m_gray[x - 1, y + 1] + 2f * m_gray[x, y + 1] + m_gray[x + 1, y + 1];
                    
                    float gradMag2 = gx * gx + gy * gy;

                    m_edgeMap[x, y] = gradMag2;
                    if (gradMag2 > maxEdgeEnergy) {
                        maxEdgeEnergy = gradMag2;
                    }
                }
            }

            if (maxEdgeEnergy <= 0f) return;

            NormalizeEdgeMap(maxEdgeEnergy, width, height, minX, maxX, minY, maxY);

            m_sEdgeMapMarker.End();
        }

        private void NormalizeEdgeMap(float maxEdgeEnergy, int width, int height, int minX, int maxX, int minY,
            int maxY) {
            float invMax = 1f / maxEdgeEnergy;

            // normalization
            for (int y = Mathf.Max(1, minY); y < Mathf.Min(height - 1, maxY); y++) {
                for (int x = Mathf.Max(1, minX); x < Mathf.Min(width - 1, maxX); x++) {
                    float normalized = m_edgeMap[x, y] * invMax;

                    // Threshold weak edges
                    m_edgeMap[x, y] = normalized >= EdgeThreshold ? normalized : 0f;
                }
            }
        }

        private void ComputeGrayscaleMask(NativeArray<Color32> sourcePixels, int width, int minX, int maxX, int minY,
            int maxY) {
            for (int y = minY; y < maxY; y++) {
                for (int x = minX; x < maxX; x++) {
                    Color c = sourcePixels[y * width + x];
                    m_gray[x, y] = c.grayscale;
                }
            }
        }
        
        

        private List<Vector2> EvolveBalloonSnake(List<Vector2> points, float[,] edgeMap, int width, int height,
            int iterations) {
            m_sEvolveSnakeMarker.Begin();
            
            int n = points.Count;

            // Gradient Gaussian Field for stability
            ComputeGradientField(edgeMap, width, height);

            for (int iter = 0; iter < iterations; iter++) {
                var totalMovement = CalculateNewContourPoints(points, n, width, height, out List<Vector2> newPoints);

                points = newPoints;

                // Check for stabilization - stop early if snake has stabilized
                float avgMovement = totalMovement / n;
                if (avgMovement < StabilizationThreshold) {
                    break;
                }

                // Redistribute points every 10 iterations to maintain uniform spacing
                if (iter % 10 == 0 && iter > 0) {
                    points = TextureUtility.RedistributePoints(points, n);
                }
            }

            m_sEvolveSnakeMarker.End();
            return points;
        }

        private float CalculateNewContourPoints(List<Vector2> points, int n, int width, int height,
            out List<Vector2> newPoints) {
            newPoints = new List<Vector2>(n);
            float totalMovement = 0f;

            for (int i = 0; i < n; i++) {
                Vector2 p = points[i];
                Vector2 prev = points[(i - 1 + n) % n];
                Vector2 next = points[(i + 1) % n];

                // Internal forces (smoothness constraints)
                Vector2 elasticity = Elasticity * (prev + next - 2 * p);

                Vector2 prevPrev = points[(i - 2 + n) % n];
                Vector2 nextNext = points[(i + 2) % n];
                Vector2 curvature = Rigidity * (prevPrev - 2 * prev + 2 * next - nextNext);

                Vector2 tangent = (next - prev).normalized;
                var newX = -tangent.y;
                var newY = tangent.x;
                Vector2 normal = new Vector2(newX, newY); // 90° rotation

                // External forces from image
                int x = Mathf.Clamp((int)p.x, 0, width - 1);
                int y = Mathf.Clamp((int)p.y, 0, height - 1);

                Vector2 grad = m_gradientField[x, y];
                float edgeStrength = m_edgeMap[x, y];

                float gradMag = grad.magnitude;
                Vector2 edgeNormal = gradMag > MinGradMagnitude ? grad / gradMag : Vector2.zero;

                // Balloon
                float flatness = Mathf.Clamp01(1f - edgeStrength);
                float dampening = Mathf.Clamp01(flatness * flatness);
                float alignment = gradMag > MinGradMagnitude - 5f ? Mathf.Max(0f, Vector2.Dot(normal, edgeNormal)) : 0f;

                Vector2 balloonForce = (edgeStrength > EdgeThreshold)
                    ? Vector2.zero
                    : MovementPerFrame * dampening * alignment * normal;

                // Edge attraction
                Vector2 edgeForce = EdgeAttraction * grad;

                // Barrier to avoid overstepping edges
                Vector2 barrierForce = -BarrierWeight * edgeStrength * grad;

                Vector2 force = elasticity + curvature + balloonForce + edgeForce + barrierForce;

                // Directional lock
                float edgeDamping = Mathf.Exp(-EdgeDampenFactor * edgeStrength);

                // Velocity
                Vector2 velocity = PositionScaling * edgeDamping * force;

                // Remove cross-edge motion
                if (edgeStrength > EdgeThreshold && gradMag > MinGradMagnitude) {
                    float normalMotion = Vector2.Dot(velocity, edgeNormal);
                    velocity -= normalMotion * edgeNormal;
                }

                var newP = p + velocity;

                // Clamp to image bounds
                newP.x = Mathf.Clamp(newP.x, 0, width - 1);
                newP.y = Mathf.Clamp(newP.y, 0, height - 1);

                // Track movement for stabilization detection
                totalMovement += Vector2.Distance(p, newP);

                newPoints.Add(newP);
            }

            return totalMovement;
        }

        private void ComputeGradientField(float[,] edgeMap, int width, int height) {
            // Compute gradient of edge map
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    float gx = (edgeMap[x + 1, y] - edgeMap[x - 1, y]) * 0.5f;
                    float gy = (edgeMap[x, y + 1] - edgeMap[x, y - 1]) * 0.5f;
                    m_gradientField[x, y] = new Vector2(gx, gy);
                }
            }
        }
    }
}