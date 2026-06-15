using UnityEngine;
using UnityEngine.Rendering;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // GPU-resident counterpart to FastMarchingAlgorithm. The exact heap-based CPU version is
    // left untouched as the baseline; this one solves the same problem entirely on the GPU:
    //   - distance field via parallel Godunov/Jacobi eikonal relaxation
    //   - colour fill via FMM band ordering (one dispatch per distance band)
    // Benchmarking is kept always-on but never blocks: pixels are pulled back with
    // AsyncGPUReadback and fed to the existing Burst evaluators in the background.
    public class FastMarchingComputeAlgorithm : AbstractGpuInpaintingAlgorithm {
        // Keep consistent with FastMarchingAlgorithm.DownscaleFactor.
        private const int DownscaleFactor = 6;
        private const string ShaderResourcePath = "Shaders/FastMarchingCompute";

        private readonly ComputeShader m_cs;
        private readonly int m_kInit, m_kEikonal, m_kFillBand, m_kPromoteBand, m_kComposite;

        // Persistent GPU resources (allocated once per resolution, reused every frame).
        private int m_fullW, m_fullH, m_dsW, m_dsH;
        private ComputeBuffer m_distance;
        private ComputeBuffer m_flags;
        private ComputeBuffer m_working;
        private RenderTexture m_result;

        public FastMarchingComputeAlgorithm() : base("FastMarchingCompute.Inpaint") {
            m_cs = Resources.Load<ComputeShader>(ShaderResourcePath);
            if (m_cs == null) {
                Debug.LogError($"[FMM-GPU] Compute shader not found at Resources/{ShaderResourcePath}.compute");
                return;
            }
            m_kInit = m_cs.FindKernel("Init");
            m_kEikonal = m_cs.FindKernel("Eikonal");
            m_kFillBand = m_cs.FindKernel("FillBand");
            m_kPromoteBand = m_cs.FindKernel("PromoteBand");
            m_kComposite = m_cs.FindKernel("Composite");
        }

        public override RenderTexture Inpaint(RenderTexture source, RectInt maskBounds) {
            if (m_cs == null || source == null) return source;

            EnsureResources(source.width, source.height);

            // Mask bounds in downscaled space.
            int dsx0 = Mathf.Clamp(maskBounds.xMin / DownscaleFactor, 0, m_dsW);
            int dsy0 = Mathf.Clamp(maskBounds.yMin / DownscaleFactor, 0, m_dsH);
            int dsx1 = Mathf.Clamp((maskBounds.xMax + DownscaleFactor - 1) / DownscaleFactor, 0, m_dsW);
            int dsy1 = Mathf.Clamp((maskBounds.yMax + DownscaleFactor - 1) / DownscaleFactor, 0, m_dsH);
            int dsMaskW = Mathf.Max(0, dsx1 - dsx0);
            int dsMaskH = Mathf.Max(0, dsy1 - dsy0);
            if (dsMaskW == 0 || dsMaskH == 0) {
                Graphics.Blit(source, m_result);
                return m_result;
            }

            // Distance info travels ~1 px/iteration; max distance inside a rect ~ min(w,h)/2.
            int maxRadius = (Mathf.Min(dsMaskW, dsMaskH) + 1) / 2 + 2;
            int eikonalIters = maxRadius + 2;

            // Shared parameters.
            m_cs.SetInts("MaskBoundsDs", dsx0, dsy0, dsx1, dsy1);
            m_cs.SetInts("MaskBoundsFull", maskBounds.xMin, maskBounds.yMin, maskBounds.xMax, maskBounds.yMax);
            m_cs.SetInt("DsW", m_dsW);
            m_cs.SetInt("DsH", m_dsH);
            m_cs.SetInt("FullW", m_fullW);
            m_cs.SetInt("FullH", m_fullH);
            m_cs.SetInt("DownscaleFactor", DownscaleFactor);

            BindAll(m_kInit, source);
            BindAll(m_kEikonal, source);
            BindAll(m_kFillBand, source);
            BindAll(m_kPromoteBand, source);
            BindAll(m_kComposite, source);

            int dsGx = Mathf.CeilToInt(m_dsW / 8f), dsGy = Mathf.CeilToInt(m_dsH / 8f);
            int fGx = Mathf.CeilToInt(m_fullW / 8f), fGy = Mathf.CeilToInt(m_fullH / 8f);

            BeginProfile();
            m_cs.Dispatch(m_kInit, dsGx, dsGy, 1);
            for (int i = 0; i < eikonalIters; i++) {
                m_cs.Dispatch(m_kEikonal, dsGx, dsGy, 1);
            }
            for (int band = 1; band <= maxRadius; band++) {
                m_cs.SetInt("CurrentBand", band);
                m_cs.Dispatch(m_kFillBand, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kPromoteBand, dsGx, dsGy, 1);
            }
            m_cs.Dispatch(m_kComposite, fGx, fGy, 1);
            EndProfileAndReport(maskBounds);

            MaybeRunEvaluation(source, m_result, maskBounds);
            return m_result;
        }

        private void BindAll(int kernel, RenderTexture source) {
            m_cs.SetTexture(kernel, "Source", source);
            m_cs.SetBuffer(kernel, "Distance", m_distance);
            m_cs.SetBuffer(kernel, "Flags", m_flags);
            m_cs.SetBuffer(kernel, "Working", m_working);
            m_cs.SetTexture(kernel, "Result", m_result);
        }

        private void EnsureResources(int fullW, int fullH) {
            if (m_result != null && m_fullW == fullW && m_fullH == fullH) return;

            ReleaseResources();

            m_fullW = fullW;
            m_fullH = fullH;
            m_dsW = Mathf.Max(1, fullW / DownscaleFactor);
            m_dsH = Mathf.Max(1, fullH / DownscaleFactor);
            int count = m_dsW * m_dsH;

            m_distance = new ComputeBuffer(count, sizeof(float));
            m_flags = new ComputeBuffer(count, sizeof(uint));
            m_working = new ComputeBuffer(count, sizeof(float) * 4);

            m_result = new RenderTexture(fullW, fullH, 0, RenderTextureFormat.ARGB32) {
                enableRandomWrite = true
            };
            m_result.Create();
        }

        private void ReleaseResources() {
            m_distance?.Release();
            m_flags?.Release();
            m_working?.Release();
            if (m_result != null) m_result.Release();
            m_distance = null;
            m_flags = null;
            m_working = null;
            m_result = null;
        }

        public override void Dispose() {
            base.Dispose();
            ReleaseResources();
        }
    }
}
