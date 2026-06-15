using UnityEngine;
using UnityEngine.Rendering;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // GPU-resident counterpart to NltmAlgorithm. The exact Burst CPU version is left untouched
    // as the baseline; this one runs the same greedy exemplar-based scheme entirely on the GPU.
    // NLTM's outer loop is inherently serial (one highest-priority patch filled per iteration),
    // so it is driven here from C# - exactly like FastMarchingComputeAlgorithm dispatches one
    // band per iteration. The per-iteration work (priority map, candidate search, trimmed-mean
    // fill, confidence update) lives in NltmCompute.compute; the heavy candidate search is the
    // part that actually benefits from the GPU.
    //
    // No data is read back inside the loop except a tiny "remaining unknown pixels" counter,
    // polled every CheckStride iterations to terminate early (the loop is otherwise bounded by
    // the downscaled mask area). Benchmarking is always-on but non-blocking via AsyncGPUReadback.
    public class NltmComputeAlgorithm : AbstractGpuInpaintingAlgorithm {
        // Keep consistent with NltmAlgorithm.
        private const int DownscaleFactor = 6;
        private const int PatchRadius = 4;
        private const int PatchArea = (2 * PatchRadius + 1) * (2 * PatchRadius + 1);
        private const float GaussianSigma = PatchRadius / 2.0f;
        private const string ShaderResourcePath = "Shaders/NltmCompute";

        // How often to poll the remaining-pixel counter (a GPU->CPU sync) to break the loop.
        private const int CheckStride = 8;

        private readonly ComputeShader m_cs;
        private readonly int m_kInit, m_kMarkCached, m_kPriority, m_kResetIter,
            m_kArgmaxMax, m_kArgmaxPick, m_kSearch, m_kSelectK, m_kFill, m_kUpdate, m_kComposite;

        // Persistent GPU resources (allocated once per resolution, reused every frame).
        private int m_fullW, m_fullH, m_dsW, m_dsH;
        private ComputeBuffer m_mask, m_confidence, m_working, m_searchAllowed, m_prio;
        private ComputeBuffer m_gaussianWeights;
        private ComputeBuffer m_targetIdx, m_maxPriority, m_validCount, m_candCount;
        private ComputeBuffer m_validIdx, m_validDist, m_candIdx, m_patchConf, m_fillColor;
        private ComputeBuffer m_newCacheCount, m_remaining;
        // Two cache buffers swapped each frame: read = previous frame, write = this frame.
        private ComputeBuffer m_cacheRead, m_cacheWrite;
        private RenderTexture m_result;

        // Temporal cache state (PatchMatch-inspired, mirrors NltmAlgorithm's cache).
        private int m_cachedCount;
        private int m_prevMaskArea = -1;

        private readonly uint[] m_remainingReadback = new uint[1];
        private readonly uint[] m_newCacheCountReadback = new uint[1];

        public NltmComputeAlgorithm() : base("NltmCompute.Inpaint") {
            m_cs = Resources.Load<ComputeShader>(ShaderResourcePath);
            if (m_cs == null) {
                Debug.LogError($"[NLTM-GPU] Compute shader not found at Resources/{ShaderResourcePath}.compute");
                return;
            }
            m_kInit = m_cs.FindKernel("Init");
            m_kMarkCached = m_cs.FindKernel("MarkCached");
            m_kPriority = m_cs.FindKernel("Priority");
            m_kResetIter = m_cs.FindKernel("ResetIter");
            m_kArgmaxMax = m_cs.FindKernel("ArgmaxMax");
            m_kArgmaxPick = m_cs.FindKernel("ArgmaxPick");
            m_kSearch = m_cs.FindKernel("Search");
            m_kSelectK = m_cs.FindKernel("SelectK");
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

            // Cache invalidation: drop the temporal cache if the mask changed by >= 10% (matches CPU).
            bool useCached = m_cachedCount > 0;
            if (m_prevMaskArea > 0) {
                float changeRatio = Mathf.Abs(dsMaskArea - m_prevMaskArea) / (float)m_prevMaskArea;
                if (changeRatio >= 0.1f) {
                    useCached = false;
                    m_cachedCount = 0;
                }
            }
            m_prevMaskArea = dsMaskArea;

            // Shared parameters (constant across the iteration loop).
            m_cs.SetInts("MaskBoundsDs", dsx0, dsy0, dsx1, dsy1);
            m_cs.SetInts("MaskBoundsFull", maskBounds.xMin, maskBounds.yMin, maskBounds.xMax, maskBounds.yMax);
            m_cs.SetInt("DsW", m_dsW);
            m_cs.SetInt("DsH", m_dsH);
            m_cs.SetInt("FullW", m_fullW);
            m_cs.SetInt("FullH", m_fullH);
            m_cs.SetInt("DownscaleFactor", DownscaleFactor);
            m_cs.SetInt("UseCached", useCached ? 1 : 0);
            m_cs.SetInt("CachedCount", useCached ? m_cachedCount : 0);

            BindAll(m_kInit, source);
            BindAll(m_kMarkCached, source);
            BindAll(m_kPriority, source);
            BindAll(m_kResetIter, source);
            BindAll(m_kArgmaxMax, source);
            BindAll(m_kArgmaxPick, source);
            BindAll(m_kSearch, source);
            BindAll(m_kSelectK, source);
            BindAll(m_kFill, source);
            BindAll(m_kUpdate, source);
            BindAll(m_kComposite, source);

            int dsGx = Mathf.CeilToInt(m_dsW / 8f), dsGy = Mathf.CeilToInt(m_dsH / 8f);
            int fGx = Mathf.CeilToInt(m_fullW / 8f), fGy = Mathf.CeilToInt(m_fullH / 8f);

            BeginProfile();
            m_cs.Dispatch(m_kInit, dsGx, dsGy, 1);
            if (useCached) {
                int cacheGroups = Mathf.CeilToInt(m_cachedCount / 64f);
                m_cs.Dispatch(m_kMarkCached, cacheGroups, 1, 1);
            }

            // Greedy fill loop. Upper bound = downscaled mask area (each iteration clears >= 1 pixel);
            // in practice a patch clears many pixels, so we poll the remaining counter to break early.
            for (int iter = 0; iter < dsMaskArea; iter++) {
                m_cs.Dispatch(m_kResetIter, 1, 1, 1);
                m_cs.Dispatch(m_kPriority, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kArgmaxMax, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kArgmaxPick, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kSearch, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kSelectK, 1, 1, 1);
                m_cs.Dispatch(m_kFill, dsGx, dsGy, 1);
                m_cs.Dispatch(m_kUpdate, dsGx, dsGy, 1);

                if ((iter & (CheckStride - 1)) == CheckStride - 1) {
                    m_remaining.GetData(m_remainingReadback);
                    if (m_remainingReadback[0] == 0) break;
                }
            }

            m_cs.Dispatch(m_kComposite, fGx, fGy, 1);
            EndProfileAndReport(maskBounds);

            // Promote this frame's chosen candidates to next frame's cache.
            m_newCacheCount.GetData(m_newCacheCountReadback);
            m_cachedCount = Mathf.Min((int)m_newCacheCountReadback[0], m_dsW * m_dsH);
            (m_cacheRead, m_cacheWrite) = (m_cacheWrite, m_cacheRead);

            MaybeRunEvaluation(source, m_result, maskBounds);
            return m_result;
        }

        private void BindAll(int kernel, RenderTexture source) {
            m_cs.SetTexture(kernel, "Source", source);
            m_cs.SetTexture(kernel, "Result", m_result);
            m_cs.SetBuffer(kernel, "Mask", m_mask);
            m_cs.SetBuffer(kernel, "Confidence", m_confidence);
            m_cs.SetBuffer(kernel, "Working", m_working);
            m_cs.SetBuffer(kernel, "SearchAllowed", m_searchAllowed);
            m_cs.SetBuffer(kernel, "Prio", m_prio);
            m_cs.SetBuffer(kernel, "GaussianWeights", m_gaussianWeights);
            m_cs.SetBuffer(kernel, "CachedCandidates", m_cacheRead);
            m_cs.SetBuffer(kernel, "TargetIdx", m_targetIdx);
            m_cs.SetBuffer(kernel, "MaxPriority", m_maxPriority);
            m_cs.SetBuffer(kernel, "ValidIdx", m_validIdx);
            m_cs.SetBuffer(kernel, "ValidDist", m_validDist);
            m_cs.SetBuffer(kernel, "ValidCount", m_validCount);
            m_cs.SetBuffer(kernel, "CandIdx", m_candIdx);
            m_cs.SetBuffer(kernel, "CandCount", m_candCount);
            m_cs.SetBuffer(kernel, "PatchConf", m_patchConf);
            m_cs.SetBuffer(kernel, "FillColor", m_fillColor);
            m_cs.SetBuffer(kernel, "NewCache", m_cacheWrite);
            m_cs.SetBuffer(kernel, "NewCacheCount", m_newCacheCount);
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
            m_searchAllowed = new ComputeBuffer(count, sizeof(uint));
            m_prio = new ComputeBuffer(count, sizeof(float));
            m_validIdx = new ComputeBuffer(count, sizeof(int));
            m_validDist = new ComputeBuffer(count, sizeof(float));
            m_cacheRead = new ComputeBuffer(count, sizeof(int));
            m_cacheWrite = new ComputeBuffer(count, sizeof(int));

            m_targetIdx = new ComputeBuffer(1, sizeof(int));
            m_maxPriority = new ComputeBuffer(1, sizeof(uint));
            m_validCount = new ComputeBuffer(1, sizeof(uint));
            m_candCount = new ComputeBuffer(1, sizeof(uint));
            m_candIdx = new ComputeBuffer(5, sizeof(int));   // K = 5
            m_patchConf = new ComputeBuffer(1, sizeof(float));
            m_fillColor = new ComputeBuffer(1, sizeof(float) * 4);
            m_newCacheCount = new ComputeBuffer(1, sizeof(uint));
            m_remaining = new ComputeBuffer(1, sizeof(uint));

            m_gaussianWeights = new ComputeBuffer(PatchArea, sizeof(float));
            m_gaussianWeights.SetData(BuildGaussianWeights());

            m_cachedCount = 0;
            m_prevMaskArea = -1;

            m_result = new RenderTexture(fullW, fullH, 0, RenderTextureFormat.ARGB32) {
                enableRandomWrite = true
            };
            m_result.Create();
        }

        // Port of PrecomputeGaussianWeights.
        private static float[] BuildGaussianWeights() {
            var weights = new float[PatchArea];
            float sigma2 = 2f * GaussianSigma * GaussianSigma;
            int idx = 0;
            for (int dy = -PatchRadius; dy <= PatchRadius; dy++) {
                for (int dx = -PatchRadius; dx <= PatchRadius; dx++) {
                    float distSq = dx * dx + dy * dy;
                    weights[idx++] = Mathf.Exp(-distSq / sigma2);
                }
            }
            return weights;
        }

        private void ReleaseResources() {
            m_mask?.Release();
            m_confidence?.Release();
            m_working?.Release();
            m_searchAllowed?.Release();
            m_prio?.Release();
            m_validIdx?.Release();
            m_validDist?.Release();
            m_cacheRead?.Release();
            m_cacheWrite?.Release();
            m_targetIdx?.Release();
            m_maxPriority?.Release();
            m_validCount?.Release();
            m_candCount?.Release();
            m_candIdx?.Release();
            m_patchConf?.Release();
            m_fillColor?.Release();
            m_newCacheCount?.Release();
            m_remaining?.Release();
            m_gaussianWeights?.Release();
            if (m_result != null) m_result.Release();

            m_mask = m_confidence = m_working = m_searchAllowed = m_prio = null;
            m_validIdx = m_validDist = m_cacheRead = m_cacheWrite = null;
            m_targetIdx = m_maxPriority = m_validCount = m_candCount = null;
            m_candIdx = m_patchConf = m_fillColor = m_newCacheCount = m_remaining = null;
            m_gaussianWeights = null;
            m_result = null;
        }

        public override void Dispose() {
            base.Dispose();
            ReleaseResources();
        }
    }
}
