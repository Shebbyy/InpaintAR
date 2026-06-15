using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace InpaintAR.Scripts.Benchmarking.Evaluators {
    public class PerformanceEvaluator : MonoBehaviour {
        private static readonly List<double> InpaintingTimeMSValues = new();
        private static int _totalInpaintedPixels;

        public static double GetTotalFPS() {
            return (1.0 / Time.deltaTime);
        }

        public static double GetAverageInpaintingTime() {
            return InpaintingTimeMSValues.Count > 0 ? InpaintingTimeMSValues.Average() : -1;
        }

        public static double GetInpaintingIsolatedFPS() {
            var avgTime = GetAverageInpaintingTime();
            return (int)avgTime != -1 ? 1.0 / (avgTime / 1000) : -1;
        }

        public static double GetAverageTimePerPixel() {
            return _totalInpaintedPixels > 0 ? InpaintingTimeMSValues.Sum() / _totalInpaintedPixels : -1;
        }

        public static void AddInpaintingStats(int inpaintedPixels, double elapsedTimeMs) {
            _totalInpaintedPixels += inpaintedPixels;
            InpaintingTimeMSValues.Add(elapsedTimeMs);
        }

        public static void ResetValues() {
            InpaintingTimeMSValues.Clear();
            _totalInpaintedPixels = 0;
        }
    }
}