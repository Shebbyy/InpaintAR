using System;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Rendering;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // GPU-resident counterpart to LaMaAlgorithm. The CPU version is left untouched as the baseline.
    //
    // Unlike the NLTM/ELTM ports there is no hand-written kernel that re-implements the algorithm:
    // LaMa is an ONNX network, and Sentis lowers it to compute shaders itself once the worker uses
    // a GPU backend. The "conversion to GPU" therefore consists of:
    //   1. running inference on BackendType.GPUCompute (the CPU baseline used BackendType.CPU),
    //   2. removing every CPU round-trip (ReadbackAndClone / ReadPixels / GetPixels32 / per-pixel
    //      loops) so the source stays on the GPU and the result is a RenderTexture,
    //   3. doing the input/output pixel glue on the GPU via TextureConverter + LaMaPrePost.compute.
    //
    // The network I/O contract (two inputs "image" and "mask", NCHW, masked pixels zeroed, output
    // in [0,255]) is preserved exactly from LaMaAlgorithm. Benchmarking is always-on but
    // non-blocking via AsyncGPUReadback.
    //
    // NOTE: a few Sentis 2.1 specifics should be confirmed in the editor against the actual model:
    // the input tensor names, that ToTensor yields NCHW [1,C,H,W], the texture Y-orientation of
    // the ToTensor/RenderToTexture round-trip, and that the output range is [0,255].
    public class LaMaComputeAlgorithm : AbstractGpuInpaintingAlgorithm {
        private const int ModelSize = 256;
        private const int MaskDilation = 2;   // matches LaMaAlgorithm.DilateMask(2) for a rect mask
        private const string ModelResourcePath = "Models/lama";
        private const string ShaderResourcePath = "Shaders/LaMaPrePost";

        private readonly Model m_model;
        private Worker m_worker;
        private bool m_ready;

        private readonly ComputeShader m_cs;
        private readonly int m_kBuildInput = -1, m_kNormalize = -1;

        // Persistent GPU resources.
        private int m_fullW, m_fullH;
        private RenderTexture m_inputRT;     // 256x256 ARGB32, masked-zeroed RGB
        private RenderTexture m_maskRT;      // 256x256 RFloat, mask channel
        private RenderTexture m_outputFloat; // full-res ARGBFloat, network output (Sentis-upscaled)
        private RenderTexture m_result;      // full-res ARGB32, normalized result

        public LaMaComputeAlgorithm() : base("LaMaCompute.Inpaint") {
            m_cs = Resources.Load<ComputeShader>(ShaderResourcePath);
            if (m_cs == null) {
                Debug.LogError($"[LaMa-GPU] Compute shader not found at Resources/{ShaderResourcePath}.compute");
                return;
            }
            m_kBuildInput = m_cs.FindKernel("BuildInput");
            m_kNormalize = m_cs.FindKernel("Normalize");

            try {
                var modelAsset = Resources.Load<ModelAsset>(ModelResourcePath);
                if (modelAsset == null) {
                    Debug.LogError($"[LaMa-GPU] Model asset not found at Resources/{ModelResourcePath}");
                    return;
                }
                m_model = ModelLoader.Load(modelAsset);
                // The whole point of the GPU conversion: run the network on the GPU compute backend.
                m_worker = new Worker(m_model, BackendType.GPUCompute);
                m_ready = true;
            }
            catch (Exception e) {
                Debug.LogError($"[LaMa-GPU] Failed to initialize model: {e.Message}");
                m_ready = false;
            }
        }

        public override RenderTexture Inpaint(RenderTexture source, RectInt maskBounds) {
            if (!m_ready || m_cs == null || source == null) return source;

            EnsureResources(source.width, source.height);

            // Mask rect mapped into 256-space, expanded by MaskDilation (dilating a solid rect ==
            // growing it), clamped to the model grid.
            float sx = (float)ModelSize / m_fullW;
            float sy = (float)ModelSize / m_fullH;
            int rx0 = Mathf.Clamp(Mathf.RoundToInt(maskBounds.xMin * sx) - MaskDilation, 0, ModelSize);
            int ry0 = Mathf.Clamp(Mathf.RoundToInt(maskBounds.yMin * sy) - MaskDilation, 0, ModelSize);
            int rx1 = Mathf.Clamp(Mathf.RoundToInt(maskBounds.xMax * sx) + MaskDilation, 0, ModelSize);
            int ry1 = Mathf.Clamp(Mathf.RoundToInt(maskBounds.yMax * sy) + MaskDilation, 0, ModelSize);
            if (rx1 <= rx0 || ry1 <= ry0) {
                Graphics.Blit(source, m_result);
                return m_result;
            }

            BeginProfile();

            // Step 1: build the masked-zeroed input image + mask channel on the GPU.
            m_cs.SetInts("MaskRect256", rx0, ry0, rx1, ry1);
            m_cs.SetInt("FullW", m_fullW);
            m_cs.SetInt("FullH", m_fullH);
            m_cs.SetTexture(m_kBuildInput, "Source", source);
            m_cs.SetTexture(m_kBuildInput, "InputTex", m_inputRT);
            m_cs.SetTexture(m_kBuildInput, "MaskTex", m_maskRT);
            int g256 = Mathf.CeilToInt(ModelSize / 8f);
            m_cs.Dispatch(m_kBuildInput, g256, g256, 1);

            // Step 2: textures -> tensors (GPU-resident), NCHW [1,C,256,256].
            using var imageTensor = TextureConverter.ToTensor(m_inputRT, ModelSize, ModelSize, 3);
            using var maskTensor = TextureConverter.ToTensor(m_maskRT, ModelSize, ModelSize, 1);

            // Step 3: inference on the GPU. No ReadbackAndClone - the output stays on the GPU.
            m_worker.SetInput("image", imageTensor);
            m_worker.SetInput("mask", maskTensor);
            m_worker.Schedule();

            if (m_worker.PeekOutput() is not Tensor<float> output) {
                Debug.LogError("[LaMa-GPU] Failed to get output tensor from worker");
                EndProfile();
                Graphics.Blit(source, m_result);
                return m_result;
            }

            // Step 4: tensor -> full-res float texture (Sentis bilinear upscale), then normalize.
            TextureConverter.RenderToTexture(output, m_outputFloat);
            m_cs.SetTexture(m_kNormalize, "OutputFloat", m_outputFloat);
            m_cs.SetTexture(m_kNormalize, "Result", m_result);
            m_cs.Dispatch(m_kNormalize, Mathf.CeilToInt(m_fullW / 8f), Mathf.CeilToInt(m_fullH / 8f), 1);
            EndProfileAndReport(maskBounds);

            MaybeRunEvaluation(source, m_result, maskBounds);
            return m_result;
        }

        private void EnsureResources(int fullW, int fullH) {
            if (m_result != null && m_fullW == fullW && m_fullH == fullH) return;

            ReleaseResources();

            m_fullW = fullW;
            m_fullH = fullH;

            m_inputRT = new RenderTexture(ModelSize, ModelSize, 0, RenderTextureFormat.ARGB32) {
                enableRandomWrite = true
            };
            m_inputRT.Create();

            m_maskRT = new RenderTexture(ModelSize, ModelSize, 0, RenderTextureFormat.RFloat) {
                enableRandomWrite = true
            };
            m_maskRT.Create();

            // Float format so the network's [0,255] output is preserved (an 8-bit RT would clip).
            m_outputFloat = new RenderTexture(fullW, fullH, 0, RenderTextureFormat.ARGBFloat) {
                enableRandomWrite = true
            };
            m_outputFloat.Create();

            // Linear (non-sRGB): UAV writes do not sRGB-encode, so a default sRGB RT would be
            // double-decoded on display and look too dark. See GpuInpaintingAdapter.
            m_result = new RenderTexture(fullW, fullH, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear) {
                enableRandomWrite = true
            };
            m_result.Create();
        }

        private void ReleaseResources() {
            if (m_inputRT != null) m_inputRT.Release();
            if (m_maskRT != null) m_maskRT.Release();
            if (m_outputFloat != null) m_outputFloat.Release();
            if (m_result != null) m_result.Release();
            m_inputRT = m_maskRT = m_outputFloat = m_result = null;
        }

        public override void Dispose() {
            base.Dispose();
            m_worker?.Dispose();
            m_worker = null;
            ReleaseResources();
            m_ready = false;
        }
    }
}
