using System;
using System.Diagnostics;
using UnityEngine;

namespace InpaintAR.Scripts.Benchmarking {
    // Measures the GPU execution time of a compute driver's dispatch block, in milliseconds.
    //
    // ComputeShader.Dispatch is asynchronous, so a plain Stopwatch around the dispatches would only
    // capture CPU submission time and under-report GPU cost. Instead we force the GPU to finish the
    // submitted work before stopping the timer: a synchronous ComputeBuffer.GetData stalls the calling
    // thread until every preceding GPU command has completed (the GPU executes in submission order, so
    // a readback issued after the dispatches can only return once they are done). The elapsed
    // wall-clock time then includes the actual GPU execution.
    //
    // This blocks the main thread for the duration of one inpaint - acceptable here: it is a
    // benchmarking build, the timings need to be comparable per algorithm, and the exemplar drivers
    // (ELTM/NLTM) already sync mid-loop anyway. We read a tiny dedicated dummy buffer rather than the
    // result texture so the sync itself stays cheap and adds no meaningful transfer cost to the sample.
    //
    // An earlier version bracketed the dispatches with CommandBuffer.BeginSample/EndSample + a GPU
    // ProfilerRecorder; that could not stay balanced on-device because the mid-loop readbacks split the
    // begin/end markers across render-thread frame boundaries (and GpuRecorder was rejected outright on
    // some backends). The Stopwatch + sync approach is uniform and robust across all four drivers.
    public sealed class GpuInpaintProfiler : IDisposable {
        private readonly Stopwatch m_watch = new();
        private readonly ComputeBuffer m_syncBuffer = new(1, sizeof(uint));
        private readonly uint[] m_syncReadback = new uint[1];
        private double m_lastMs = -1;

        // markerName is no longer used for measurement; kept for API parity / future logging.
        public GpuInpaintProfiler(string markerName) { }

        // Call immediately before the first Dispatch.
        public void Begin() {
            m_watch.Reset();
            m_watch.Start();
        }

        // Call immediately after the last Dispatch. The GetData readback stalls until all preceding
        // GPU work has completed, so the stopwatch captures GPU execution and not just submission.
        public void End() {
            m_syncBuffer.GetData(m_syncReadback);
            m_watch.Stop();
            m_lastMs = m_watch.Elapsed.TotalMilliseconds;
        }

        // Last measured GPU time in milliseconds, or -1 before the first completed sample.
        public double LastMilliseconds => m_lastMs;

        public void Dispose() {
            m_syncBuffer?.Release();
        }
    }
}