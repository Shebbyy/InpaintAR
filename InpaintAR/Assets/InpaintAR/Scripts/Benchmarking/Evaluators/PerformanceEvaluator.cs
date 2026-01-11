using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace InpaintAR.Scripts.Benchmarking.Evaluators {
    public class PerformanceEvaluator : MonoBehaviour {
        private static readonly List<long> InpaintingTimeMSValues = new();
        private static int _totalInpaintedPixels = 0;

        public static double GetTotalFPS() {
            return (1.0 / Time.smoothDeltaTime);
        }

        public static double GetAverageInpaintingTime() {
            return InpaintingTimeMSValues.Average() / 1000;
        }

        public static double GetInpaintingIsolatedFPS() {
            return 1.0 / GetAverageInpaintingTime();
        }

        public static double GetAverageTimePerPixel() {
            return (double)InpaintingTimeMSValues.Sum() / 1000 / _totalInpaintedPixels;
        }

        public static void AddInpaintingStats(int inpaintedPixels, long elapsedTimeMs) {
            _totalInpaintedPixels += inpaintedPixels;
            InpaintingTimeMSValues.Add(elapsedTimeMs);
        }

        public static void ResetValues() {
            InpaintingTimeMSValues.Clear();
            _totalInpaintedPixels = 0;
        }
    }
}