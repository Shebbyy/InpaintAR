using System;
using System.Collections.Generic;
using UnityEngine;
using InpaintAR.Scripts.Util;
using Unity.Sentis;

namespace InpaintAR.Scripts.Inpainting.Algorithms {
    // LaMa (Large Mask Inpainting) Algorithm implementation using Unity Sentis.
    // Performs deep learning-based inpainting at a fixed resolution
    public class LaMaAlgorithm : AbstractInpaintingAlgorithm, IDisposable {
        private const int ModelSize = 256;
        // Model must be placed in Assets/Resources/Models/lama.onnx
        // Unity will import it as a ModelAsset automatically
        private const string ModelResourcePath = "Models/lama";

        // Persistent resources (loaded once)
        private Model m_model;
        private Worker m_worker;
        private bool m_isInitialized;
        private bool m_isDisposed;

        // Pre-allocated arrays for tensor data (reusable)
        private float[] m_imageData;  // [3 * ModelSize * ModelSize] for RGB
        private float[] m_maskData;   // [ModelSize * ModelSize] for single channel

        // Pre-allocated arrays for texture conversion
        private Color32[] m_downsampledPixels;
        private Color32[] m_outputPixels;

        // Cached textures for resampling
        private Texture2D m_downsampledTexture;
        private RenderTexture m_downsampleRT;
        private RenderTexture m_upsampleRT;

        public LaMaAlgorithm() {
            Initialize();
        }

        // Initialize the LaMa model and pre-allocate resources.
        private void Initialize() {
            if (m_isInitialized) return;

            try {
                // Load the model asset from Resources folder
                var modelAsset = Resources.Load<ModelAsset>(ModelResourcePath);

                if (modelAsset == null) {
                    Debug.LogError($"[LaMa] Model asset not found at Resources/{ModelResourcePath}");
                    Debug.LogError("[LaMa] Please place lama.onnx in Assets/Resources/Models/");
                    return;
                }

                // Load the model from the asset
                m_model = ModelLoader.Load(modelAsset);

                // Create worker with GPU compute backend for Quest optimization
                m_worker = new Worker(m_model, BackendType.GPUCompute);

                // Pre-allocate tensors
                PreAllocateResources();

                m_isInitialized = true;
                Debug.Log("[LaMa] Model initialized successfully from Resources");
            }
            catch (Exception e) {
                Debug.LogError($"[LaMa] Failed to initialize model: {e.Message}");
                m_isInitialized = false;
            }
        }

        private void PreAllocateResources() {
            int pixelCount = ModelSize * ModelSize;

            // Pre-allocate arrays for tensor data
            m_imageData = new float[3 * pixelCount];  // RGB channels
            m_maskData = new float[pixelCount];        // Single channel

            // Pre-allocate arrays for texture conversion
            m_downsampledPixels = new Color32[pixelCount];
            m_outputPixels = new Color32[pixelCount];

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
                // Step 1: Downsample source texture to ModelSize x ModelSize
                DownsampleTexture(source);

                // Step 2: Convert mask indices to downsampled coordinates
                PrepareMaskData(maskPixelIndices, originalWidth, originalHeight);

                // Step 3: Prepare image data from downsampled pixels
                PrepareImageData();

                // Step 4: Run inference (creates fresh tensors from arrays)
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
        }

        private void PrepareMaskData(HashSet<int> maskPixelIndices, int originalWidth, int originalHeight) {
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
                // Data is in NCHW layout [1, 1, H, W] flattened
                int maskIndex = downY * ModelSize + downX;
                m_maskData[maskIndex] = 1f;
            }

            // Dilate mask slightly to ensure coverage after downsampling
            DilateMask(2);
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

        private void PrepareImageData() {
            // Convert Color32[] to float array in NCHW format
            // Model expects RGB in [0, 1] range with masked regions zeroed out
            // Layout: [1, 3, ModelSize, ModelSize] flattened = [R channel][G channel][B channel]
            int channelSize = ModelSize * ModelSize;

            for (int y = 0; y < ModelSize; y++) {
                for (int x = 0; x < ModelSize; x++) {
                    int pixelIdx = y * ModelSize + x;
                    Color32 pixel = m_downsampledPixels[pixelIdx];

                    // Check if this pixel is in the mask (to be inpainted)
                    bool isMasked = m_maskData[pixelIdx] > 0f;

                    // LaMa expects masked regions to be zeroed in the input image
                    // NCHW layout: channel 0 (R), channel 1 (G), channel 2 (B)
                    // Normalize to [0, 1] range
                    if (isMasked) {
                        m_imageData[0 * channelSize + pixelIdx] = 0f;
                        m_imageData[1 * channelSize + pixelIdx] = 0f;
                        m_imageData[2 * channelSize + pixelIdx] = 0f;
                    }
                    else {
                        m_imageData[0 * channelSize + pixelIdx] = pixel.r / 255f;
                        m_imageData[1 * channelSize + pixelIdx] = pixel.g / 255f;
                        m_imageData[2 * channelSize + pixelIdx] = pixel.b / 255f;
                    }
                }
            }
        }

        private Tensor<float> RunInference() {
            // Create fresh tensors from pre-filled arrays
            // Tensors become GPU-bound after inference and can't be rewritten
            var imageShape = new TensorShape(1, 3, ModelSize, ModelSize);
            var maskShape = new TensorShape(1, 1, ModelSize, ModelSize);

            using var imageTensor = new Tensor<float>(imageShape, m_imageData);
            using var maskTensor = new Tensor<float>(maskShape, m_maskData);

            // Log model input/output info on first run for debugging
            LogModelInfo();

            // Set inputs - LaMa typically uses "image" and "mask" as input names
            m_worker.SetInput("image", imageTensor);
            m_worker.SetInput("mask", maskTensor);

            // Execute inference
            m_worker.Schedule();

            // Get output tensor - typically named "output"
            var gpuTensor = m_worker.PeekOutput() as Tensor<float>;

            if (gpuTensor == null) {
                Debug.LogError("[LaMa] Failed to get output tensor from worker");
                return null;
            }

            // Download from GPU to CPU - ReadbackAndClone returns a new CPU-readable tensor
            var cpuTensor = gpuTensor.ReadbackAndClone();

            return cpuTensor;
        }

        private bool m_hasLoggedModelInfo;

        private void LogModelInfo() {
            if (m_hasLoggedModelInfo || m_model == null) return;
            m_hasLoggedModelInfo = true;

            Debug.Log("[LaMa] === Model Input/Output Info ===");

            // Log inputs
            foreach (var input in m_model.inputs) {
                Debug.Log($"[LaMa] Input: name='{input.name}', shape={input.shape}");
            }

            // Log outputs
            foreach (var output in m_model.outputs) {
                Debug.Log($"[LaMa] Output: name='{output.name}'");
            }
        }

        private bool m_hasLoggedOutputRange;

        private Texture2D ProcessOutput(Tensor<float> outputTensor, int originalWidth, int originalHeight) {
            if (outputTensor == null) {
                Debug.LogError("[LaMa] Output tensor is null");
                return new Texture2D(originalWidth, originalHeight, TextureFormat.RGBA32, false);
            }

            // Debug: Log output tensor shape and value range on first run
            if (!m_hasLoggedOutputRange) {
                m_hasLoggedOutputRange = true;
                float minVal = float.MaxValue, maxVal = float.MinValue;
                for (int c = 0; c < 3; c++) {
                    for (int y = 0; y < ModelSize; y++) {
                        for (int x = 0; x < ModelSize; x++) {
                            float val = outputTensor[0, c, y, x];
                            minVal = Mathf.Min(minVal, val);
                            maxVal = Mathf.Max(maxVal, val);
                        }
                    }
                }
                Debug.Log($"[LaMa] Output tensor shape: {outputTensor.shape}");
                Debug.Log($"[LaMa] Output value range: min={minVal}, max={maxVal}");

                // Log a few sample values
                Debug.Log($"[LaMa] Sample output[0,0,128,128]: R={outputTensor[0, 0, 128, 128]}, G={outputTensor[0, 1, 128, 128]}, B={outputTensor[0, 2, 128, 128]}");
            }

            // Convert NCHW tensor to Color32 array
            // Model outputs values in [0, 255] range
            for (int y = 0; y < ModelSize; y++) {
                for (int x = 0; x < ModelSize; x++) {
                    int pixelIdx = y * ModelSize + x;

                    m_outputPixels[pixelIdx] = new Color32(
                        (byte)Mathf.Clamp(outputTensor[0, 0, y, x], 0, 255),
                        (byte)Mathf.Clamp(outputTensor[0, 1, y, x], 0, 255),
                        (byte)Mathf.Clamp(outputTensor[0, 2, y, x], 0, 255),
                        255
                    );
                }
            }

            var outputTexture = new Texture2D(ModelSize, ModelSize, TextureFormat.RGBA32, false);
            outputTexture.SetPixels32(m_outputPixels);
            outputTexture.Apply();

            // Upscale to original resolution
            var result = UpsampleTexture(outputTexture, originalWidth, originalHeight);

            MPixelBuffer = result.GetPixels32();

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

            return result;
        }

        public void Dispose() {
            if (m_isDisposed) return;

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
