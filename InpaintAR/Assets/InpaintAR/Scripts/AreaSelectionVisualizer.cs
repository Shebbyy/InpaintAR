using System;
using System.Collections.Generic;
using InpaintAR.Scripts.Inpainting;
using JetBrains.Annotations;
using Meta.XR;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;

namespace InpaintAR.Scripts {
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

        private RectTransform FillRectMask { get; set; }

        // Internal references to the generated UI elements
        private RectTransform m_leftCornerBox;
        private RectTransform m_rightCornerBox;
        private Canvas m_canvas;

        private RawImage FillImage { get; set; }

        [Header("Debug Settings")] 
        [Tooltip("Uses a solid red box instead of the camera texture for display")]
        public bool showDebugRect = true;

        private Ray m_textureTopLeft;
        private Ray m_textureBottomRight;
        private PassthroughCameraAccess m_cameraAccess;
        private RectTransform m_fillImageRect; // rect transform of the RawImage which contains the inpainted content
        private Camera m_mainCam;
        [CanBeNull] private Texture2D m_copiedTexture; // Copy of the passthrough texture for inpainting, etc.
        [CanBeNull] private HashSet<int> m_inpaintMask;
        [CanBeNull] private Texture2D m_inpaintedTexture;
        private IInpaintingAlgorithm m_inpaintingAlgorithm;

        private void Start() {
            if (!areaDetection) {
                throw new Exception("AreaDetection Object is required!");
            }

            CreateUICanvasAndCorners();

            InitializeCamera();
            
            m_inpaintingAlgorithm = InpaintingFactory.GetInpaintingAlgorithm(inpaintingAlgorithmSelection);
        }

        private void InitializeCamera() {
            m_cameraAccess = gameObject.AddComponent<PassthroughCameraAccess>();
            m_cameraAccess.CameraPosition = PassthroughCameraAccess.CameraPositionType.Left;
            m_cameraAccess.RequestedResolution = new Vector2Int(1280, 960);
            m_mainCam = Camera.main;
        }
        
        private void OnDestroy() {
            if (m_copiedTexture) {
                Destroy(m_copiedTexture);
            }
        }

        void Update() {
            if (!areaDetection
                || !m_leftCornerBox
                || !m_rightCornerBox) return;

            UpdateCorner(
                m_leftCornerBox,
                areaDetection.LeftHandCornerScreenPos
            );

            UpdateCorner(
                m_rightCornerBox,
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
                
                FillRectMask.gameObject.SetActive(true);
                // Update canvas position to follow camera
                UpdateCanvasWorldPosition();

                UpdateSelectionMaskPosition(areaDetection.LeftHandCornerScreenPos.Value, areaDetection.RightHandCornerScreenPos.Value);
            }
            else {
                // Update to only adjust to camera angle
                UpdateCanvasWorldPosition(false);
            }
            
            if (   showDebugRect
                || !IsSelectionAreaWithinCameraView()) return;

            UpdateImage(isSelectionActive);
            
        }

        private void UpdateImage(bool isSelectionActive) {
            Texture sourceTexture = m_cameraAccess.GetTexture();
            if (sourceTexture) {
                m_copiedTexture = CopyTexture(sourceTexture);
            }
                
            UpdatePassthroughImagePosition();

            // During Selection no Edge-Detection/Inpainting
            if (isSelectionActive) {
                FillImage.texture = m_copiedTexture;
                return;
            }
            
            m_inpaintMask = SnakeEdgeDetection.GetContourMaskPixelIndices(FillImage.rectTransform, m_copiedTexture, FillRectMask);
            m_inpaintedTexture = m_inpaintingAlgorithm.Inpaint(m_copiedTexture, m_inpaintMask);
            
            FillImage.texture = m_inpaintedTexture;
        }

        private void UpdateSelectionMaskPosition(Vector2 leftHandCornerScreenPos, Vector2 rightHandCornerScreenPos) {
            // Selection Rectangle Calculation
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                (RectTransform)m_canvas.transform,
                leftHandCornerScreenPos,
                m_mainCam,
                out Vector2 selLeftLocal
            );

            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                (RectTransform)m_canvas.transform,
                rightHandCornerScreenPos,
                m_mainCam,
                out Vector2 selRightLocal
            );

            Vector2 selectionBottomLeft = new Vector2(Mathf.Min(selLeftLocal.x, selRightLocal.x),
                Mathf.Min(selLeftLocal.y, selRightLocal.y));
            Vector2 selectionTopRight = new Vector2(Mathf.Max(selLeftLocal.x, selRightLocal.x),
                Mathf.Max(selLeftLocal.y, selRightLocal.y));

            // Mask Update
            FillRectMask.localPosition = selectionBottomLeft;
            FillRectMask.sizeDelta = selectionTopRight - selectionBottomLeft;
        }

        private void UpdatePassthroughImagePosition() {
            // Image position calculation so it matches with the passthrough background as close as possible
            m_textureTopLeft = m_cameraAccess.ViewportPointToRay(new Vector2(0, 0));
            m_textureBottomRight = m_cameraAccess.ViewportPointToRay(new Vector2(1, 1));

            float distance = GetCameraDistance();
            Vector3 topLeftWorldPos = m_textureTopLeft.origin + m_textureTopLeft.direction * distance;
            Vector3 bottomRightWorldPos = m_textureBottomRight.origin + m_textureBottomRight.direction * distance;
            Vector3 topLeftScreenPos = m_mainCam.WorldToScreenPoint(topLeftWorldPos);
            Vector3 bottomRightScreenPos = m_mainCam.WorldToScreenPoint(bottomRightWorldPos);

            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                (RectTransform)m_canvas.transform,
                topLeftScreenPos,
                m_mainCam,
                out Vector2 topLeftLocalPos
            );

            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                (RectTransform)m_canvas.transform,
                bottomRightScreenPos,
                m_mainCam,
                out Vector2 bottomRightLocalPos
            );
            
            Vector2 bottomLeft = new Vector2(Mathf.Min(topLeftLocalPos.x, bottomRightLocalPos.x), Mathf.Min(topLeftLocalPos.y, bottomRightLocalPos.y));
            Vector2 topRight = new Vector2(Mathf.Max(topLeftLocalPos.x, bottomRightLocalPos.x), Mathf.Max(topLeftLocalPos.y, bottomRightLocalPos.y));
            
            // Readjust Child Image Position to adjust for FillRect Mask Update
            m_fillImageRect.localPosition = bottomLeft - (Vector2)FillRectMask.localPosition;
            m_fillImageRect.sizeDelta = topRight - bottomLeft;
        }

        private void UpdateCorner(RectTransform cornerElem, Vector2? screenPos) {
            if (!screenPos.HasValue) {
                cornerElem.gameObject.SetActive(false);
                return;
            }

            cornerElem.gameObject.SetActive(true);

            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                (RectTransform)cornerElem.parent,
                screenPos.Value,
                m_mainCam,
                out Vector2 localPos
            );

            cornerElem.localPosition = localPos;
        }

        private void UpdateCanvasWorldPosition(bool withPosition = true) {
            if (!m_canvas || !m_mainCam) return;
            
            // Position canvas at far clip plane of camera
            Transform cam = m_mainCam.transform;
            float distance = GetCameraDistance();
            if (withPosition) {
                m_canvas.transform.position = cam.position + cam.forward * distance;
            }
            // Make canvas always face the camera (billboard like)
            m_canvas.transform.rotation = cam.rotation;
        }

        private void CreateUICanvasAndCorners() {
            GameObject canvasObj = new GameObject("AreaSelectionUICanvas");
            m_canvas = canvasObj.AddComponent<Canvas>();
            m_canvas.renderMode = RenderMode.WorldSpace;
            
            // Ensure canvas renders on top of passthrough background to avoid z-buffer fighting/clipping
            m_canvas.sortingOrder = 500;
            
            // Position canvas in world space
            RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
            canvasRect.sizeDelta = new Vector2(canvasWidth * pixelsPerUnit, canvasHeight * pixelsPerUnit);
            
            float scale = 1f / pixelsPerUnit;
            canvasObj.transform.localScale = new Vector3(scale, scale, scale);
            
            // Position in front of camera (will be updated each frame)
            UpdateCanvasWorldPosition();

            CanvasScaler scaler = canvasObj.AddComponent<CanvasScaler>();
            scaler.dynamicPixelsPerUnit = pixelsPerUnit;

            canvasObj.AddComponent<GraphicRaycaster>();

            // Left Corner Icons for Selection
            GameObject leftObj = new GameObject("LeftCornerUI");
            leftObj.transform.SetParent(canvasObj.transform, false);
            m_leftCornerBox = leftObj.AddComponent<RectTransform>();
            SetupCornerImage(m_leftCornerBox, true);
            leftObj.SetActive(false);

            GameObject rightObj = new GameObject("RightCornerUI");
            rightObj.transform.SetParent(canvasObj.transform, false);
            m_rightCornerBox = rightObj.AddComponent<RectTransform>();
            SetupCornerImage(m_rightCornerBox, false);
            rightObj.SetActive(false);

            // Mask Container
            GameObject fillObj = new GameObject("MaskContainer");
            fillObj.transform.SetParent(canvasObj.transform, false);
            FillRectMask = fillObj.AddComponent<RectTransform>();

            FillRectMask.pivot = new Vector2(0, 0);
            FillRectMask.anchorMin = FillRectMask.anchorMax = new Vector2(0.5f, 0.5f);

            // Adding RectMask2D; FillImage is clipped off, only showing content of FillObj
            fillObj.AddComponent<RectMask2D>();

            // Image Child Item -> RectTransform to allow for individual placement separate from FillRect Mask
            GameObject fillImageObj = new GameObject("FillImage");
            fillImageObj.transform.SetParent(fillObj.transform, false);
            m_fillImageRect = fillImageObj.AddComponent<RectTransform>();
            m_fillImageRect.anchorMin = m_fillImageRect.anchorMax = new Vector2(0, 0);
            m_fillImageRect.pivot = new Vector2(0, 0);

            FillImage = fillImageObj.AddComponent<RawImage>();
            FillImage.color = showDebugRect ? Color.red : Color.white;

            // Start hidden
            FillRectMask.gameObject.SetActive(false);
        }

        private void SetupCornerImage(RectTransform target, bool isLeft) {
            var img = target.GetComponent<Image>();
            if (!img) {
                img = target.gameObject.AddComponent<Image>();
            }

            if (!img.sprite) {
                img.sprite = GenerateCornerSprite(isLeft);
            }

            float anchorX = isLeft ? 0 : 1;
            float anchorY = isLeft ? 1 : 0;

            target.anchorMin = target.anchorMax = new Vector2(anchorX, anchorY); // center canvas to camera
            target.pivot = new Vector2(anchorX, anchorY);
            target.sizeDelta = new Vector2(cornerSpriteSize, cornerSpriteSize);
        }

        private Sprite GenerateCornerSprite(bool isLeft) {
            int size = cornerSpriteSize;
            int thickness = cornerSpriteThickness;
            Texture2D texture = new Texture2D(size, size);

            Color[] pixels = new Color[size * size];
            for (int i = 0; i < pixels.Length; i++) {
                pixels[i] = Color.clear;
            }

            texture.SetPixels(pixels);

            // Horizontal line
            int yStart = isLeft ? size - thickness : 0;
            int yEnd = isLeft ? size : thickness;
            for (int x = 0; x < size; x++) {
                for (int y = yStart; y < yEnd; y++) {
                    texture.SetPixel(x, y, Color.red);
                }
            }

            // Vertical Line
            int xStart = isLeft ? 0 : size - thickness;
            int xEnd = isLeft ? thickness : size;

            for (int x = xStart; x < xEnd; x++) {
                for (int y = 0; y < size; y++) {
                    texture.SetPixel(x, y, Color.red);
                }
            }

            texture.Apply();
            return Sprite.Create(texture, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f));
        }

        private float GetCameraDistance() {
            // use max dimension of canvas to avoid clipping due to rotation, etc. of canvas
            return m_mainCam.farClipPlane - 5f;
        }

        private bool IsSelectionAreaWithinCameraView() {
            if (!FillRectMask || !FillRectMask.gameObject.activeSelf) {
                return false;
            }

            // Get the four corners of the selection rectangle in world space
            Vector3[] corners = new Vector3[4];
            FillRectMask.GetWorldCorners(corners);

            // Check if all corners are within the camera's frustum and viewport
            foreach (Vector3 corner in corners) {
                Vector3 viewportPoint = m_mainCam.WorldToViewportPoint(corner);
                
                // Check if the point is in front of the camera and within viewport bounds
                if (viewportPoint.z <= 0 || viewportPoint.x < 0 || viewportPoint.x > 1 || 
                    viewportPoint.y < 0 || viewportPoint.y > 1) {
                    return false;
                }
            }

            return true;
        }

        private Texture2D CopyTexture(Texture source) {
            if (m_copiedTexture) {
                Destroy(m_copiedTexture);
            }
            m_copiedTexture = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);

            RenderTexture currentRT = RenderTexture.active;

            // Source is usually RenderTexture when delivered from Quest
            RenderTexture texture = source as RenderTexture;
            RenderTexture.active = texture;
            
            m_copiedTexture.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
            m_copiedTexture.Apply();
            
            RenderTexture.active = currentRT;
            
            return m_copiedTexture;
        }
    }
}