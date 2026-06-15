using UnityEngine;
using UnityEngine.Rendering;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // GPU-resident counterpart to EltmAlgorithm (Deng-Huang-Zhao 2014). The exact Burst CPU
    // version is left untouched as the baseline; this one runs the same greedy exemplar scheme
    // entirely on the GPU. ELTM's outer loop is inherently serial (one highest-priority patch
    // filled per iteration), so it is driven here from C# - exactly like FastMarchingCompute and
    // NltmCompute. The per-iteration work (priority map, local-window SSD search, single-source
    // fill, regularized confidence update) lives in EltmCompute.compute.
    //
    // Unlike the NLTM port this has no temporal cache and a single best match (no top-K / no
    // trimmed mean), so it is a lighter pipeline. No data is read back inside the loop except a
    // tiny "remaining unknown pixels" counter, polled every CheckStride iterations to terminate
    // early. Benchmarking is always-on but non-blocking via AsyncGPUReadback.
    public class EltmComputeAlgorithm : AbstractGpuInpaintingAlgorithm {
        // Keep consistent with EltmAlgorithm.
        private const int DownscaleFactor = 6;
        private const string ShaderResourcePath = "Shaders/EltmCompute";

        // How often to poll the remaining-pixel counter (a GPU->CPU sync) to break the loop.
        private const int CheckStride = 8;

        private readonly ComputeShader m_cs;
        private readonly int m_kInit, m_kPriority, m_kResetIter, m_kArgmaxMax, m_kArgmaxPick,
            m_kSearch, m_kSearchPick, m_kPrep, m_kFill, m_kUpdate, m_kComposite;

        // Persistent GPU resources (allocated once per resolution, reused every frame).
        private int m_fullW, m_fullH, m_dsW, m_dsH;
        private ComputeBuffer m_mask, m_confidence, m_working, m_prio, m_dist;
        private ComputeBuffer m_targetIdx, m_maxPriority, m_bestDist, m_bestPatch;
        private ComputeBuffer m_regConf, m_fillColor, m_remaining;
        private RenderTexture m_result;

        private readonly uint[] m_remainingReadback = new uint[1];

        public EltmComputeAlgorithm() : base("EltmCompute.Inpaint") {
            m_cs = Resources.Load<ComputeShader>(ShaderResourcePath);
            if (m_cs == null) {
                Debug.LogError($"[ELTM-GPU] Compute shader not found at Resources/{ShaderResourcePath}.compute");
                return;
            }
            m_kInit = m_cs.FindKernel("Init");
            m_kPriority = m_cs.FindKernel("Priority");
            m_kResetIter = m_cs.FindKernel("ResetIter");
            m_kArgmaxMax = m_cs.FindKernel("ArgmaxMax");
            m_kArgmaxPick = m_cs.FindKernel("ArgmaxPick");
            m_kSearch = m_cs.FindKernel("Search");
            m_kSearchPick = m_cs.FindKernel("SearchPick");
            m_kPrep = m_cs.FindKernel("Prep");
            m_kFill = m_cs.FindKernel("Fill");
            m_kUpdate = m_cs.FindKernel("Update");
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
            int dsMaskArea = dsMaskW * dsMaskH;
            if (dsMaskArea == 0) {
                Graphics.Blit(source, m_result);
                return m_result;
            }

            // Shared parameters (constant across the iteration loop).
            m_cs.SetInts("MaskBoundsDs", dsx0, dsy0, dsx1, dsy1);
            m_cs.SetInts("MaskBoundsFull", maskBounds.xMin, maskBounds.yMin, maskBounds.xMax, maskBounds.yMax);
            m_cs.SetInt("DsW", m_dsW);
            m_cs.SetInt("DsH", m_dsH);
            m_cs.SetInt("FullW", m_fullW);
            m_cs.SetInt("FullH", m_fullH);
            m_cs.SetInt("DownscaleFactor", DownscaleFactor);

            BindAll(m_kInit, source);
            BindAll(m_kPriority, source);
            BindAll(m_kResetIter, source);
            BindAll(m_kArgmaxMax, source);
            BindAll(m_kArgmaxPick, source);
            BindAll(m_kSearch, source);
            BindAll(m_kSearchPick, source);
            BindAll(m_kPrep, source);
            BindAll(m_kFill, source);
            BindAll(m_kUpdate, source);
            BindAll(m_kComposite, source);

            int dsGx = Mathf.CeilToInt(m_dsW / 8f), dsGy = Mathf.CeilToInt(m_dsH / 8f);
            int fGx = Mathf.CeilToInt(m_fullW / 8f), fGy = Mathf.CeilToInt(m_fullH / 8f);

            BeginProfile();
            m_cs.Dispatch(m_kInit, dsGx, dsGy, 1);

            // Greedy fill loop. Upper bound = downscaled mask area (each iteration clears >= 1 pixel);
            // in practice a patch clears many pixels, so we poll the remaining counter to break early.
            for (int iter = 0; iter < dsMaskArea; iter++) {
                m_cs.Dispatch(m_kResetIter, 1, 1, 1);
                m_cs.Dispatch(m_kPriority, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kArgmaxMax, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kArgmaxPick, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kSearch, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kSearchPick, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kPrep, 1, 1, 1);
                m_cs.Dispatch(m_kFill, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kUpdate, dsGx, dsGy, 1);

                if ((iter & (CheckStride - 1)) == CheckStride - 1) {
                    m_remaining.GetData(m_remainingReadback);
                    if (m_remainingReadback[0] == 0) break;
                }
            }

            m_cs.Dispatch(m_kComposite, fGx, fGy, 1);
            EndProfileAndReport(maskBounds);

            MaybeRunEvaluation(source, m_result, maskBounds);
            return m_result;
        }

        private void BindAll(int kernel, RenderTexture source) {
            m_cs.SetTexture(kernel, "Source", source);
            m_cs.SetTexture(kernel, "Result", m_result);
            m_cs.SetBuffer(kernel, "Mask", m_mask);
            m_cs.SetBuffer(kernel, "Confidence", m_confidence);
            m_cs.SetBuffer(kernel, "Working", m_working);
            m_cs.SetBuffer(kernel, "Prio", m_prio);
            m_cs.SetBuffer(kernel, "Dist", m_dist);
            m_cs.SetBuffer(kernel, "TargetIdx", m_targetIdx);
            m_cs.SetBuffer(kernel, "MaxPriority", m_maxPriority);
            m_cs.SetBuffer(kernel, "BestDist", m_bestDist);
            m_cs.SetBuffer(kernel, "BestPatch", m_bestPatch);
            m_cs.SetBuffer(kernel, "RegConf", m_regConf);
            m_cs.SetBuffer(kernel, "FillColor", m_fillColor);
            m_cs.SetBuffer(kernel, "Remaining", m_remaining);
        }

        private void EnsureResources(int fullW, int fullH) {
            if (m_result != null && m_fullW == fullW && m_fullH == fullH) return;

            ReleaseResources();

            m_fullW = fullW;
            m_fullH = fullH;
            m_dsW = Mathf.Max(1, fullW / DownscaleFactor);
            m_dsH = Mathf.Max(1, fullH / DownscaleFactor);
            int count = m_dsW * m_dsH;

            m_mask = new ComputeBuffer(count, sizeof(uint));
            m_confidence = new ComputeBuffer(count, sizeof(float));
            m_working = new ComputeBuffer(count, sizeof(float) * 4);
            m_prio = new ComputeBuffer(count, sizeof(float));
            m_dist = new ComputeBuffer(count, sizeof(float));

            m_targetIdx = new ComputeBuffer(1, sizeof(int));
            m_maxPriority = new ComputeBuffer(1, sizeof(uint));
            m_bestDist = new ComputeBuffer(1, sizeof(uint));
            m_bestPatch = new ComputeBuffer(1, sizeof(int));
            m_regConf = new ComputeBuffer(1, sizeof(float));
            m_fillColor = new ComputeBuffer(1, sizeof(float) * 4);
            m_remaining = new ComputeBuffer(1, sizeof(uint));

            m_result = new RenderTexture(fullW, fullH, 0, RenderTextureFormat.ARGB32) {
                enableRandomWrite = true
            };
            m_result.Create();
        }

        private void ReleaseResources() {
            m_mask?.Release();
            m_confidence?.Release();
            m_working?.Release();
            m_prio?.Release();
            m_dist?.Release();
            m_targetIdx?.Release();
            m_maxPriority?.Release();
            m_bestDist?.Release();
            m_bestPatch?.Release();
            m_regConf?.Release();
            m_fillColor?.Release();
            m_remaining?.Release();
            if (m_result != null) m_result.Release();

            m_mask = m_confidence = m_working = m_prio = m_dist = null;
            m_targetIdx = m_maxPriority = m_bestDist = m_bestPatch = null;
            m_regConf = m_fillColor = m_remaining = null;
            m_result = null;
        }

        public override void Dispose() {
            base.Dispose();
            ReleaseResources();
        }
    }
}
