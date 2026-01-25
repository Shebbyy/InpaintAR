using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using InpaintAR.Scripts.Util;
using Unity.Sentis;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // LaMa (Large Mask Inpainting) Algorithm implementation using Unity Sentis.
    // Performs deep learning-based inpainting at 256x256 resolution
    public class LaMaAlgorithm : AbstractInpaintingAlgorithm, IDisposable {
        private const int ModelSize = 256;
        private const string ModelFileName = "lama.onnx";

        // Persistent resources (loaded once)
        private Model m_model;
        private Worker m_worker;
        private bool m_isInitialized;
        private bool m_isDisposed;

        // Pre-allocated tensors for reuse (avoid GC)
        private Tensor<float> m_inputImageTensor;
        private Tensor<float> m_inputMaskTensor;

        // Pre-allocated arrays for texture conversion
        private Color32[] m_downsampledPixels;
        private float[] m_maskData;
        private Color32[] m_outputPixels;

        // Cached textures for resampling
        private Texture2D m_downsampledTexture;
        private RenderTexture m_downsampleRT;
        private RenderTexture m_upsampleRT;

        public LaMaAlgorithm() {
            Initialize();
        }

        // Initialize the LaMa model and pre-allocate resources.
        // Called automatically on construction.
        private void Initialize() {
            if (m_isInitialized) return;

            try {
                string modelPath = Path.Combine(Application.streamingAssetsPath, "Models", ModelFileName);

                if (!File.Exists(modelPath)) {
                    Debug.LogError($"[LaMa] Model file not found at: {modelPath}");
                    Debug.LogError("[LaMa] Please download lama_256.onnx from https://huggingface.co/Carve/LaMa-ONNX");
                    return;
                }

                // Load the ONNX model
                m_model = ModelLoader.Load(modelPath);

                // Create worker with GPU compute backend for Quest optimization
                m_worker = new Worker(m_model, BackendType.GPUCompute);

                // Pre-allocate tensors
                PreAllocateResources();

                m_isInitialized = true;
                Debug.Log("[LaMa] Model initialized successfully with GPU compute backend");
            }
            catch (Exception e) {
                Debug.LogError($"[LaMa] Failed to initialize model: {e.Message}");
                m_isInitialized = false;
            }
        }

        private void PreAllocateResources() {
            int pixelCount = ModelSize * ModelSize;

            // Pre-allocate arrays
            m_downsampledPixels = new Color32[pixelCount];
            m_maskData = new float[pixelCount]; // Single channel
            m_outputPixels = new Color32[pixelCount];

            // Pre-allocate tensors with correct shapes
            // LaMa expects: image [1, 3, 256, 256], mask [1, 1, 256, 256]
            m_inputImageTensor = new Tensor<float>(new TensorShape(1, 3, ModelSize, ModelSize));
            m_inputMaskTensor = new Tensor<float>(new TensorShape(1, 1, ModelSize, ModelSize));

            // Pre-allocate textures
            m_downsampledTexture = new Texture2D(ModelSize, ModelSize, TextureFormat.RGBA32, false);
            m_downsampleRT = new RenderTexture(ModelSize, ModelSize, 0, RenderTextureFormat.ARGB32);
            m_downsampleRT.Create();
        }

        protected override Texture2D InpaintLogic(Texture2D source, HashSet<int> maskPixelIndices) {
            if (!m_isInitialized) {
                Debug.LogWarning("[LaMa] Model not initialized, returning source texture");
                return source;
            }

            if (maskPixelIndices.Count == 0) {
                return source;
            }

            int originalWidth = TextureUtility.GetImageWidth(source);
            int originalHeight = TextureUtility.GetImageHeight(source);

            try {
                // Step 1: Downsample source texture to 256x256
                DownsampleTexture(source);

                // Step 2: Convert mask indices to downsampled coordinates and create mask tensor
                PrepareMaskTensor(maskPixelIndices, originalWidth, originalHeight);

                // Step 3: Prepare image tensor from downsampled pixels
                PrepareImageTensor();

                // Step 4: Run inference
                var outputTensor = RunInference();

                // Step 5: Convert output tensor to texture and upscale
                var result = ProcessOutput(outputTensor, originalWidth, originalHeight);

                // Dispose the output tensor
                outputTensor?.Dispose();

                return result;
            }
            catch (Exception e) {
                Debug.LogError($"[LaMa] Inference failed: {e.Message}");
                return source;
            }
        }

        private void DownsampleTexture(Texture2D source) {
            // Use GPU blit for fast downsampling
            RenderTexture.active = m_downsampleRT;
            Graphics.Blit(source, m_downsampleRT);

            // Read pixels from RenderTexture
            m_downsampledTexture.ReadPixels(new Rect(0, 0, ModelSize, ModelSize), 0, 0);
            m_downsampledTexture.Apply();

            m_downsampledPixels = m_downsampledTexture.GetPixels32();
            RenderTexture.active = null;
        }

        private void PrepareMaskTensor(HashSet<int> maskPixelIndices, int originalWidth, int originalHeight) {
            // Clear mask data
            Array.Clear(m_maskData, 0, m_maskData.Length);

            float scaleX = (float)ModelSize / originalWidth;
            float scaleY = (float)ModelSize / originalHeight;

            // Convert each mask pixel to downsampled coordinates
            foreach (int index in maskPixelIndices) {
                int origX = index % originalWidth;
                int origY = index / originalWidth;

                // Map to downsampled coordinates
                int downX = Mathf.Clamp(Mathf.RoundToInt(origX * scaleX), 0, ModelSize - 1);
                int downY = Mathf.Clamp(Mathf.RoundToInt(origY * scaleY), 0, ModelSize - 1);

                // LaMa mask format: 1 = inpaint region
                int maskIndex = downY * ModelSize + downX;
                m_maskData[maskIndex] = 1f;
            }

            // Dilate mask slightly to ensure coverage after downsampling
            DilateMask(2);

            // Upload to tensor (NCHW format: [1, 1, H, W])
            for (int y = 0; y < ModelSize; y++) {
                for (int x = 0; x < ModelSize; x++) {
                    int srcIdx = y * ModelSize + x;
                    m_inputMaskTensor[0, 0, y, x] = m_maskData[srcIdx];
                }
            }
        }

        private void DilateMask(int radius) {
            // Simple dilation to ensure mask coverage
            var tempMask = new float[m_maskData.Length];
            Array.Copy(m_maskData, tempMask, m_maskData.Length);

            for (int y = 0; y < ModelSize; y++) {
                for (int x = 0; x < ModelSize; x++) {
                    int idx = y * ModelSize + x;
                    if (tempMask[idx] > 0.5f) {
                        // Dilate to neighbors
                        for (int dy = -radius; dy <= radius; dy++) {
                            for (int dx = -radius; dx <= radius; dx++) {
                                int nx = x + dx;
                                int ny = y + dy;
                                if (nx >= 0 && nx < ModelSize && ny >= 0 && ny < ModelSize) {
                                    m_maskData[ny * ModelSize + nx] = 1f;
                                }
                            }
                        }
                    }
                }
            }
        }

        private void PrepareImageTensor() {
            // Convert Color32[] to normalized float tensor in NCHW format
            // LaMa expects RGB normalized to [0, 1]
            for (int y = 0; y < ModelSize; y++) {
                for (int x = 0; x < ModelSize; x++) {
                    int pixelIdx = y * ModelSize + x;
                    Color32 pixel = m_downsampledPixels[pixelIdx];

                    // Normalize to [0, 1]
                    m_inputImageTensor[0, 0, y, x] = pixel.r / 255f; // R
                    m_inputImageTensor[0, 1, y, x] = pixel.g / 255f; // G
                    m_inputImageTensor[0, 2, y, x] = pixel.b / 255f; // B
                }
            }
        }

        private Tensor<float> RunInference() {
            // Set inputs - LaMa typically uses "image" and "mask" as input names
            // Adjust these names based on your specific model export
            m_worker.SetInput("image", m_inputImageTensor);
            m_worker.SetInput("mask", m_inputMaskTensor);

            // Execute inference
            m_worker.Schedule();

            // Get output tensor - typically named "output"
            var outputTensor = m_worker.PeekOutput() as Tensor<float>;

            // Make readable (download from GPU)
            outputTensor?.ReadbackRequest();
            outputTensor?.ReadbackAndClone();

            return outputTensor;
        }

        private Texture2D ProcessOutput(Tensor<float> outputTensor, int originalWidth, int originalHeight) {
            if (outputTensor == null) {
                Debug.LogError("[LaMa] Output tensor is null");
                return new Texture2D(originalWidth, originalHeight, TextureFormat.RGBA32, false);
            }

            // Convert NCHW tensor to Color32 array
            for (int y = 0; y < ModelSize; y++) {
                for (int x = 0; x < ModelSize; x++) {
                    int pixelIdx = y * ModelSize + x;

                    // Clamp and convert from [0, 1] to [0, 255]
                    byte r = (byte)Mathf.Clamp(outputTensor[0, 0, y, x] * 255f, 0, 255);
                    byte g = (byte)Mathf.Clamp(outputTensor[0, 1, y, x] * 255f, 0, 255);
                    byte b = (byte)Mathf.Clamp(outputTensor[0, 2, y, x] * 255f, 0, 255);

                    m_outputPixels[pixelIdx] = new Color32(r, g, b, 255);
                }
            }

            // Create output texture at model size
            var outputTexture = new Texture2D(ModelSize, ModelSize, TextureFormat.RGBA32, false);
            outputTexture.SetPixels32(m_outputPixels);
            outputTexture.Apply();

            // Upscale to original resolution
            var result = UpsampleTexture(outputTexture, originalWidth, originalHeight);

            // Update the inpainted pixel buffer for quality evaluation
            MInpaintedPixelBuffer = result.GetPixels32();

            // Clean up intermediate texture
            UnityEngine.Object.Destroy(outputTexture);

            return result;
        }

        private Texture2D UpsampleTexture(Texture2D source, int targetWidth, int targetHeight) {
            // Use GPU blit for fast upsampling with bilinear filtering
            if (!m_upsampleRT || m_upsampleRT.width != targetWidth || m_upsampleRT.height != targetHeight) {
                if (m_upsampleRT) {
                    m_upsampleRT.Release();
                    UnityEngine.Object.Destroy(m_upsampleRT);
                }
                m_upsampleRT = new RenderTexture(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);
                m_upsampleRT.Create();
            }

            RenderTexture.active = m_upsampleRT;
            Graphics.Blit(source, m_upsampleRT);

            var result = new Texture2D(targetWidth, targetHeight, TextureFormat.RGBA32, false);
            result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
            result.Apply();

            RenderTexture.active = null;

            return result;
        }

        public void Dispose() {
            if (m_isDisposed) return;

            m_inputImageTensor?.Dispose();
            m_inputMaskTensor?.Dispose();
            m_worker?.Dispose();

            if (m_downsampleRT != null) {
                m_downsampleRT.Release();
                UnityEngine.Object.Destroy(m_downsampleRT);
            }

            if (m_upsampleRT != null) {
                m_upsampleRT.Release();
                UnityEngine.Object.Destroy(m_upsampleRT);
            }

            if (m_downsampledTexture != null) {
                UnityEngine.Object.Destroy(m_downsampledTexture);
            }

            m_isDisposed = true;
            m_isInitialized = false;

            Debug.Log("[LaMa] Resources disposed");
        }

        ~LaMaAlgorithm() {
            Dispose();
        }
    }
}
