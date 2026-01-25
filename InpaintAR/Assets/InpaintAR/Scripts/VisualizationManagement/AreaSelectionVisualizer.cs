using System;
using System.Collections.Generic;
using InpaintAR.Scripts.Benchmarking.Evaluators;
using InpaintAR.Scripts.Inpainting;
using InpaintAR.Scripts.Input;
using InpaintAR.Scripts.SnakeEdgeDetection;
using InpaintAR.Scripts.Util;
using JetBrains.Annotations;
using UnityEngine;

namespace InpaintAR.Scripts.VisualizationManagement {
    public class AreaSelectionVisualizer : MonoBehaviour {
        [Header("RequiredData")] 
        [Tooltip("Area Detection Object. If null, attempts to find it in the scene.")]
        public AreaSelectionGestureHandler areaDetection;

        [Tooltip("Size of the corner sprites image")]
        public int cornerSpriteSize = 128;

        [Tooltip("Thickness of the corner sprite lines")]
        public int cornerSpriteThickness = 50;

        [Header("World Space Canvas Settings")]
        [Tooltip("Physical width of the canvas in world units")]
        public float canvasWidth = 2.0f;
        
        [Tooltip("Physical height of the canvas in world units")]
        public float canvasHeight = 1.5f;
        
        [Tooltip("Pixels per unit for the world space canvas")]
        public float pixelsPerUnit = 1000f;

        [Header("Inpainting Settings")] 
        [Tooltip("Which algorithm to use for the inpainting")] 
        public InpaintingAlgorithms inpaintingAlgorithmSelection;

        [Header("Debug Settings")] 
        [Tooltip("Uses a solid red box instead of the camera texture for display")]
        public bool showDebugRect = true;
        
        [CanBeNull] private Texture2D m_copiedTexture; // Copy of the passthrough texture for inpainting, etc.
        [CanBeNull] private HashSet<int> m_inpaintMask;
        [CanBeNull] private Texture2D m_inpaintedTexture;
        private AbstractInpaintingAlgorithm m_abstractInpaintingAlgorithm;
        private CameraController m_cameraController;
        private SelectionUiController m_selectionUiController;

        private void Start() {
            if (!areaDetection) {
                throw new Exception("AreaDetection Object is required!");
            }

            m_cameraController = new CameraController(gameObject);
            
            m_selectionUiController = gameObject.AddComponent<SelectionUiController>();
            m_selectionUiController.SetConfig(m_cameraController, showDebugRect);
            m_selectionUiController.CreateUICanvasAndCorners(canvasWidth, canvasHeight, pixelsPerUnit, cornerSpriteSize, cornerSpriteThickness);
            
            m_abstractInpaintingAlgorithm = InpaintingFactory.GetInpaintingAlgorithm(inpaintingAlgorithmSelection);
        }
        
        private void OnDestroy() {
            if (m_copiedTexture) {
                Destroy(m_copiedTexture);
            }
            if (m_inpaintedTexture) {
                Destroy(m_inpaintedTexture);
            }

            // Dispose algorithm if it implements IDisposable (e.g., LaMa with GPU resources)
            if (m_abstractInpaintingAlgorithm is System.IDisposable disposable) {
                disposable.Dispose();
            }

            // Clean up corner sprites and their textures
            m_selectionUiController.Cleanup();
        }
        


        void Update() {
            if (!areaDetection
                || !m_selectionUiController.GetLeftCornerBox()
                || !m_selectionUiController.GetRightCornerBox()) return;

            UpdateCorner(
                m_selectionUiController.GetLeftCornerBox(),
                areaDetection.LeftHandCornerScreenPos
            );

            UpdateCorner(
                m_selectionUiController.GetRightCornerBox(),
                areaDetection.RightHandCornerScreenPos
            );

            bool isSelectionActive = false;
            // only update if rectangle is valid
            if (   areaDetection.LeftHandCornerScreenPos.HasValue
                && areaDetection.RightHandCornerScreenPos.HasValue
                && areaDetection.LeftHandCornerScreenPos.Value.x <= areaDetection.RightHandCornerScreenPos.Value.x
                && areaDetection.LeftHandCornerScreenPos.Value.y >= areaDetection.RightHandCornerScreenPos.Value.y) {
                isSelectionActive = true;
                m_inpaintMask = null;
                
                m_selectionUiController.GetFillRectMask().gameObject.SetActive(true);
                // Update canvas position to follow camera
                m_selectionUiController.UpdateCanvasWorldPosition();

                m_selectionUiController.UpdateSelectionMaskPosition(areaDetection.LeftHandCornerScreenPos.Value, areaDetection.RightHandCornerScreenPos.Value);
                SnakeController.ResetSelectionMask();
                PerformanceEvaluator.ResetValues();
                QualityEvaluator.ResetValues();
                ClutterEvaluator.ResetValues();
            }
            else {
                // Update to only adjust to camera angle
                m_selectionUiController.UpdateCanvasWorldPosition(false);
            }
            
            if (   showDebugRect
                || !m_selectionUiController.IsSelectionAreaWithinCameraView()) return;

            UpdateImage(isSelectionActive);
        }

        private void UpdateImage(bool isSelectionActive) {
            m_selectionUiController.UpdatePassthroughImagePosition();

            // During Selection no Edge-Detection/Inpainting
            if (isSelectionActive) {
                var image = m_selectionUiController.GetFillImage();
                image.texture = null;
                image.color = new Color(0, 0, 0, 0.5f); // Dark Overlay of Mask
                return;
            }

            m_selectionUiController.GetFillImage().color = Color.white;
            
            Texture sourceTexture = m_cameraController.GetPassthroughTexture();
            if (sourceTexture) {
                if (m_copiedTexture) {
                    Destroy(m_copiedTexture);
                }
                m_copiedTexture = TextureUtility.CopyTexture(sourceTexture);
            }
            
            m_inpaintMask = SnakeController.GetContourMaskPixelIndices(m_selectionUiController.GetFillImage().rectTransform, m_copiedTexture, m_selectionUiController.GetFillRectMask());
            
            // Destroy old inpainted texture before creating new one to prevent memory leak
            if (m_inpaintedTexture) {
                Destroy(m_inpaintedTexture);
            }
            m_inpaintedTexture = m_abstractInpaintingAlgorithm.Inpaint(m_copiedTexture, m_inpaintMask);
            
            m_selectionUiController.GetFillImage().texture = m_inpaintedTexture;
        }

        private void UpdateCorner(RectTransform cornerElem, Vector2? screenPos) {
            if (!screenPos.HasValue) {
                cornerElem.gameObject.SetActive(false);
                return;
            }

            cornerElem.gameObject.SetActive(true);

            cornerElem.localPosition = m_cameraController.ScreenPointToLocalPoint((RectTransform)cornerElem.parent, screenPos.Value);
        }
    }
}