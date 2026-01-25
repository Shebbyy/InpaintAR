using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace InpaintAR.Scripts.Benchmarking.Evaluators {
    public class ClutterEvaluator {
        private static readonly List<float> ClutterReductionResults = new();
        public static void EvaluateClutterReduction(Color32[] originalImage, Color32[] inpaintedImage, int width, int height,
            HashSet<int> maskPixelIndices) {
            
        }
        
        public static void ResetValues() {
            ClutterReductionResults.Clear();
        }

        public static double GetAverageClutterReduction() {
            return ClutterReductionResults.Count > 0 ? ClutterReductionResults.Average() : 0;
        }
    }
}