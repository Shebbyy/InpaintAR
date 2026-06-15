using System.Collections.Generic;
using InpaintAR.Scripts.Benchmarking;
using InpaintAR.Scripts.Benchmarking.Evaluators;
using UnityEngine;
using UnityEngine.Rendering;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // Shared scaffolding for the GPU-resident inpainting drivers, mirroring how the CPU-side
    // AbstractInpaintingAlgorithm centralizes timing + evaluation. Holds the GPU-time profiler and
    // the always-on, non-blocking quality/clutter read-back, so each concrete driver contains only
    // its own kernels, resources, and dispatch sequence.
    public abstract class AbstractGpuInpaintingAlgorithm : IGpuInpaintingAlgorithm {
        private readonly GpuInpaintProfiler m_profiler;
        private bool m_evalBusy;

        protected AbstractGpuInpaintingAlgorithm(string profilerMarker) {
            m_profiler = new GpuInpaintProfiler(profilerMarker);
        }

        // Bracket the dispatch sequence: BeginProfile() before the first Dispatch, then
        // EndProfileAndReport() after the last - it stops the GPU timer and feeds PerformanceEvaluator.
        // EndProfile() just balances the sample without reporting (e.g. on an early/error return).
        protected void BeginProfile() => m_profiler.Begin();

        protected void EndProfile() => m_profiler.End();

        protected void EndProfileAndReport(RectInt maskBounds) {
            EndProfile();
            double gpuMs = m_profiler.LastMilliseconds;
            if (gpuMs >= 0) PerformanceEvaluator.AddInpaintingStats(maskBounds.width * maskBounds.height, gpuMs);
        }

        // Always-on, non-blocking benchmarking. Self-throttles to one read-back cycle at a time,
        // exactly like the Burst evaluators skip frames while a previous job is still running.
        protected void MaybeRunEvaluation(RenderTexture source, RenderTexture result, RectInt maskBounds) {
            if (m_evalBusy) return;
            m_evalBusy = true;

            int w = result.width, h = result.height;
            AsyncGPUReadback.Request(source, 0, srcReq => {
                if (srcReq.hasError) { m_evalBusy = false; return; }
                var original = srcReq.GetData<Color32>().ToArray();

                AsyncGPUReadback.Request(result, 0, resReq => {
                    if (!resReq.hasError) {
                        var inpainted = resReq.GetData<Color32>().ToArray();
                        var maskIndices = RectToIndices(maskBounds, w, h);
                        QualityEvaluator.EvaluateQuality(original, inpainted, w, h, maskIndices);
                        ClutterEvaluator.EvaluateClutterReduction(original, inpainted, w, h, maskIndices);
                    }
                    m_evalBusy = false;
                });
            });
        }

        private static HashSet<int> RectToIndices(RectInt r, int width, int height) {
            int x0 = Mathf.Clamp(r.xMin, 0, width);
            int x1 = Mathf.Clamp(r.xMax, 0, width);
            int y0 = Mathf.Clamp(r.yMin, 0, height);
            int y1 = Mathf.Clamp(r.yMax, 0, height);
            var set = new HashSet<int>(Mathf.Max(0, (x1 - x0) * (y1 - y0)));
            for (int y = y0; y < y1; y++) {
                int row = y * width;
                for (int x = x0; x < x1; x++) set.Add(row + x);
            }
            return set;
        }

        public abstract RenderTexture Inpaint(RenderTexture source, RectInt maskBounds);

        public virtual void Dispose() {
            m_profiler.Dispose();
        }
    }
}
