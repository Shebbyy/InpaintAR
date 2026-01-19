using System.Collections.Generic;
using InpaintAR.Scripts.Util;
using UnityEngine;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    public class NltmAlgorithm : AbstractInpaintingAlgorithm {
        private const int PatchRadius = 4; // x Pixels from center of patch away
        private int m_missingPixelCount = 0;
        private readonly float m_max = 255.0f;
        private int m_width;
        private int m_height;
        
        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            m_width = TextureUtility.GetImageWidth(source);
            m_height = TextureUtility.GetImageHeight(source);
            
            m_missingPixelCount = maskPixelIndices.Count;

            return source;
        }

        protected void InpaintNltm(HashSet<int> maskPixelIndices) {
            while (m_missingPixelCount > 0) {
                var target = GetTargetPatchIndex(maskPixelIndices);
            }
        }
        
        private int GetTargetPatchIndex(HashSet<int> maskPixelIndices) {
            var contourPixels = GetContourPixels(maskPixelIndices);
            float maxPriority = -1f;
            int bestPIndex = -1;

            foreach (var contourPixel in contourPixels) {
                var p = new Vector2Int(contourPixel % m_width, contourPixel / m_width);
                var confidence = CalculateConfidence(maskPixelIndices, p);
                
                Vector2 np = EstimateBoundaryNormal(maskPixelIndices, p);
                
                float data = CalculateDataTerm(maskPixelIndices, contourPixel, p, np);
                
                var priority = confidence * data;
                if (priority <= maxPriority) continue;
                
                maxPriority = priority;
                bestPIndex = contourPixel;
            }

            return bestPIndex;
        }
        
        Vector2 EstimateBoundaryNormal(HashSet<int> maskPixelIndices, Vector2Int p) {
            // 1.0 for target, 0.0 for source
            float GetMaskVal(int x, int y) {
                Vector2Int pos = new Vector2Int(x, y);
                if (p.x < 0 || p.x >= m_width || p.y < 0 || p.y >= m_height) return 0;
                return maskPixelIndices.Contains(GetIndex(pos)) ? 1f : 0;
            }

            float dx = (GetMaskVal(p.x + 1, p.y) - GetMaskVal(p.x - 1, p.y)) / 2f;
            float dy = (GetMaskVal(p.x, p.y + 1) - GetMaskVal(p.x, p.y - 1)) / 2f;

            Vector2 normal = new Vector2(dx, dy);

            // Normalize to get a unit vector for the dot product in D(p)
            return normal.sqrMagnitude > 0.0001f ? normal.normalized : Vector2.up; // Fallback
        }
        
        float CalculateDataTerm(HashSet<int> maskPixelIndices, int index, Vector2Int p, Vector2 normal) {
            // 1. Calculate Gradient (Sobel or simple difference)
            Vector2 gradient = GetGradient(maskPixelIndices, index, p);
    
            Vector2 isophote = new Vector2(-gradient.y, gradient.x);

            float dot = Mathf.Abs(isophote.x * normal.x + isophote.y * normal.y);
    
            return dot / m_max;
        }
        
        // Returns the gradient vector (df/dx, df/dy) at pixel p
        Vector2 GetGradient(HashSet<int> maskPixelIndices, int index, Vector2Int p) {
            // Sobel Kernels
            // Gx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            // Gy: [[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]]
    
            float gx = 0, gy = 0;

            // We only sample from known pixels (mask == false)
            // If a neighbor is in the mask, we treat it as the center pixel's value
            float GetVal(int x, int y) {
                Vector2Int pos = new Vector2Int(x, y);
                if ((p.x < 0 || p.x >= m_width || p.y < 0 || p.y >= m_height) 
                    || maskPixelIndices.Contains(GetIndex(pos))) {
                    return ((Color)MSourcePixelBuffer[index]).grayscale; 
                }

                return ((Color)MSourcePixelBuffer[GetIndex(pos)]).grayscale;
            }

            gx += -1 * GetVal(p.x - 1, p.y - 1) + 1 * GetVal(p.x + 1, p.y - 1);
            gx += -2 * GetVal(p.x - 1, p.y)     + 2 * GetVal(p.x + 1, p.y);
            gx += -1 * GetVal(p.x - 1, p.y + 1) + 1 * GetVal(p.x + 1, p.y + 1);

            gy +=  1 * GetVal(p.x - 1, p.y - 1) + 2 * GetVal(p.x, p.y - 1) + 1 * GetVal(p.x + 1, p.y - 1);
            gy += -1 * GetVal(p.x - 1, p.y + 1) - 2 * GetVal(p.x, p.y + 1) - 1 * GetVal(p.x + 1, p.y + 1);

            return new Vector2(gx, gy);
        }
        
        float CalculateConfidence(HashSet<int> maskPixelIndices, Vector2 p) {
            int countKnown = 0;
            int totalPixels = 0;

            for (int y = -PatchRadius; y <= PatchRadius; y++) {
                for (int x = -PatchRadius; x <= PatchRadius; x++) {
                    Vector2 neighbor = p + new Vector2(x, y);
                    if (p.x < 0 || p.x >= m_width || p.y < 0 || p.y >= m_height) continue;
                    
                    totalPixels++;
                    if (!maskPixelIndices.Contains(GetIndex(neighbor))) {
                        countKnown++;
                    }
                }
            }
            return (float)countKnown / totalPixels;
        }

        private int GetIndex(Vector2 pixelPos) {
            return (int)pixelPos.y * m_width + (int)pixelPos.x;
        }
        
        private HashSet<int> GetContourPixels(HashSet<int> maskPixelIndices) {
            var contourPixels = new HashSet<int>();
            var directions = new[] { -1, 1, -m_width, m_width }; // Left, Right, Up, Down
            int pixelCount = m_width * m_height;
        
            foreach (var index in maskPixelIndices) {
        
                foreach (var direction in directions) {
                    int neighbor = index + direction;
        
                    // Check if the neighbor is within bounds and not part of the mask
                    if (neighbor >= 0 && neighbor < pixelCount && maskPixelIndices.Contains(neighbor)) continue;
                    contourPixels.Add(index);
                    break;
                }
            }
        
            return contourPixels;
        }
    }
}