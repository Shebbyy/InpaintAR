using Meta.XR;
using UnityEngine;
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

        public RectTransform FillRectMask { get; private set; }

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
        private RectTransform m_fillImageRect; // rect transform of the child RawImage
        private Camera m_mainCam;

        private void Start() {
            if (!areaDetection) {
                Debug.LogError("AreaSelectionVisualizer: AreaDetection not set!");
                return;
            }

            CreateUICanvasAndCorners();

            m_cameraAccess = gameObject.AddComponent<PassthroughCameraAccess>();
            m_cameraAccess.CameraPosition = PassthroughCameraAccess.CameraPositionType.Left;
            m_cameraAccess.RequestedResolution = new Vector2Int(1280, 960);
            m_mainCam = Camera.main;
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

            // only update if rectangle is valid
            if (   areaDetection.LeftHandCornerScreenPos.HasValue
                && areaDetection.RightHandCornerScreenPos.HasValue
                && areaDetection.LeftHandCornerScreenPos.Value.x <= areaDetection.RightHandCornerScreenPos.Value.x
                && areaDetection.LeftHandCornerScreenPos.Value.y >= areaDetection.RightHandCornerScreenPos.Value.y) {
                FillRectMask.gameObject.SetActive(true);
                // Update canvas position to follow camera
                UpdateCanvasWorldPosition();

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

                // Selection Rectangle Calculation
                RectTransformUtility.ScreenPointToLocalPointInRectangle(
                    (RectTransform)m_canvas.transform,
                    areaDetection.LeftHandCornerScreenPos.Value,
                    m_mainCam,
                    out Vector2 selLeftLocal
                );

                RectTransformUtility.ScreenPointToLocalPointInRectangle(
                    (RectTransform)m_canvas.transform,
                    areaDetection.RightHandCornerScreenPos.Value,
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
                
                if (!showDebugRect) {
                    FillImage.texture = m_cameraAccess.GetTexture();
                }

                // Readjust Child Image Position to adjust for FillRect Mask Update
                m_fillImageRect.localPosition = bottomLeft - selectionBottomLeft;
                m_fillImageRect.sizeDelta = topRight - bottomLeft;
            }
            else {
                // Update to only adjust to camera angle
                UpdateCanvasWorldPosition(false);
                if (!showDebugRect) {
                    FillImage.texture = m_cameraAccess.GetTexture();
                }
            }
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
            if (withPosition) {
                float distance = GetCameraDistance();
                m_canvas.transform.position = cam.position + cam.forward * distance;
            }
            m_canvas.transform.rotation = cam.rotation;
        }

        private void CreateUICanvasAndCorners() {
            GameObject canvasObj = new GameObject("AreaSelectionUICanvas");
            m_canvas = canvasObj.AddComponent<Canvas>();
            // Use World Space rendering
            m_canvas.renderMode = RenderMode.WorldSpace;
            
            // Position canvas in world space
            RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
            canvasRect.sizeDelta = new Vector2(canvasWidth * pixelsPerUnit, canvasHeight * pixelsPerUnit);
            
            // Scale down to physical size
            float scale = 1f / pixelsPerUnit;
            canvasObj.transform.localScale = new Vector3(scale, scale, scale);
            
            // Position in front of camera (will be updated each frame)
            UpdateCanvasWorldPosition();

            CanvasScaler scaler = canvasObj.AddComponent<CanvasScaler>();
            scaler.dynamicPixelsPerUnit = pixelsPerUnit;

            canvasObj.AddComponent<GraphicRaycaster>();

            // Left Corner
            GameObject leftObj = new GameObject("LeftCornerUI");
            leftObj.transform.SetParent(canvasObj.transform, false);
            m_leftCornerBox = leftObj.AddComponent<RectTransform>();
            SetupCornerImage(m_leftCornerBox, true);
            leftObj.SetActive(false);

            // Right Corner
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

            // Add a RectMask2D so children are clipped to this rectangle (window behavior)
            fillObj.AddComponent<RectMask2D>();

            // Image Child Item -> RectTransform to allow for individual placement separate from FillRect Mask
            GameObject fillImageObj = new GameObject("FillImage");
            fillImageObj.transform.SetParent(fillObj.transform, false);
            m_fillImageRect = fillImageObj.AddComponent<RectTransform>();
            m_fillImageRect.anchorMin = m_fillImageRect.anchorMax = new Vector2(0, 0);
            m_fillImageRect.pivot = new Vector2(0, 0);

            // Add a RawImage for passthrough texture
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
            target.pivot = new Vector2(anchorX, anchorY); // center reference to image center
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
            return m_mainCam.farClipPlane - 0.3f;
        }
    }
}