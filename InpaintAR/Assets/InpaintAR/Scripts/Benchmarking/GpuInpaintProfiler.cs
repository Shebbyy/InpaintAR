using System;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Rendering;

namespace InpaintAR.Scripts.Benchmarking {
    // Measures the GPU execution time of a compute driver's dispatch block, in milliseconds.
    //
    // A stopwatch around ComputeShader.Dispatch only captures CPU submission time (dispatches are
    // async), which would under-report GPU cost. Instead, two tiny command buffers emit a GPU
    // profiler sample that brackets the existing direct Dispatch calls (GPU submission order is
    // preserved, so the sample encloses them), and a ProfilerRecorder in GPU mode reads the elapsed
    // GPU time. The reading lags by ~1 frame because the GPU finishes after the CPU submits the frame
    // - irrelevant for a running average. If GPU recording is unavailable on the platform the recorder
    // is simply invalid and LastMilliseconds returns -1 (no stats are recorded).
    public sealed class GpuInpaintProfiler : IDisposable {
        private readonly CommandBuffer m_begin;
        private readonly CommandBuffer m_end;
        private ProfilerRecorder m_recorder;

        public GpuInpaintProfiler(string markerName) {
            m_begin = new CommandBuffer { name = markerName + ".Begin" };
            m_begin.BeginSample(markerName);
            m_end = new CommandBuffer { name = markerName + ".End" };
            m_end.EndSample(markerName);
            m_recorder = ProfilerRecorder.StartNew(
                ProfilerCategory.Render, markerName, 1, ProfilerRecorderOptions.GpuRecorder);
        }

        // Call immediately before the first Dispatch and immediately after the last one.
        public void Begin() => Graphics.ExecuteCommandBuffer(m_begin);
        public void End() => Graphics.ExecuteCommandBuffer(m_end);

        // Last measured GPU time in milliseconds, or -1 when no sample is available yet / unsupported.
        public double LastMilliseconds {
            get {
                if (!m_recorder.Valid) return -1;
                long ns = m_recorder.LastValue;       // ProfilerRecorder time samples are nanoseconds
                return ns > 0 ? ns * 1e-6 : -1;
            }
        }

        public void Dispose() {
            if (m_recorder.Valid) m_recorder.Dispose();
            m_begin?.Release();
            m_end?.Release();
        }
    }
}
