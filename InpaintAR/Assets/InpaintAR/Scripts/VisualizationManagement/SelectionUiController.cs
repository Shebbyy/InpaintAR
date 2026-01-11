using UnityEngine;
using UnityEngine.UI;

namespace InpaintAR.Scripts.VisualizationManagement {
    public class SelectionUiController : MonoBehaviour {
        private RectTransform m_leftCornerBox;
        private RectTransform m_rightCornerBox;
        
        
        private Canvas m_canvas;
        private RectTransform m_fillImageRect; // rect transform of the RawImage which contains the inpainted content
        private RectTransform FillRectMask { get; set; }
        private RawImage FillImage { get; set; }
        private Ray m_textureTopLeft;
        private Ray m_textureBottomRight;
        
        private bool m_showDebugRect;
        
        private CameraController m_cameraController;
        
        public void CreateUICanvasAndCorners(float canvasWidth, float canvasHeight, float pixelsPerUnit, int cornerSpriteSize, int cornerSpriteThickness) {
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
            SetupCornerImage(m_leftCornerBox, true, cornerSpriteSize, cornerSpriteThickness);
            leftObj.SetActive(false);

            GameObject rightObj = new GameObject("RightCornerUI");
            rightObj.transform.SetParent(canvasObj.transform, false);
            m_rightCornerBox = rightObj.AddComponent<RectTransform>();
            SetupCornerImage(m_rightCornerBox, false, cornerSpriteSize, cornerSpriteThickness);
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
            FillImage.color = m_showDebugRect ? Color.red : Color.white;

            // Start hidden
            FillRectMask.gameObject.SetActive(false);
        }

        public void UpdateCanvasWorldPosition(bool withPosition = true) {
            if (!m_canvas) return;
            
            // Position canvas at far clip plane of camera
            Transform cam = m_cameraController.GetMainCameraTransform();
            float distance = m_cameraController.GetCameraDistance();
            if (withPosition) {
                m_canvas.transform.position = cam.position + cam.forward * distance;
            }
            // Make canvas always face the camera (billboard like)
            m_canvas.transform.rotation = cam.rotation;
        }

        public void Cleanup() {
            CleanupCornerSprite(m_leftCornerBox);
            CleanupCornerSprite(m_rightCornerBox);
        }
        
                
        public void UpdateSelectionMaskPosition(Vector2 leftHandCornerScreenPos, Vector2 rightHandCornerScreenPos) {
            // Selection Rectangle Calculation
            Vector2 selLeftLocal = m_cameraController.ScreenPointToLocalPoint((RectTransform)m_canvas.transform, leftHandCornerScreenPos);
            Vector2 selRightLocal = m_cameraController.ScreenPointToLocalPoint((RectTransform)m_canvas.transform, rightHandCornerScreenPos);
            
            Vector2 selectionBottomLeft = new Vector2(Mathf.Min(selLeftLocal.x, selRightLocal.x),
                Mathf.Min(selLeftLocal.y, selRightLocal.y));
            Vector2 selectionTopRight = new Vector2(Mathf.Max(selLeftLocal.x, selRightLocal.x),
                Mathf.Max(selLeftLocal.y, selRightLocal.y));

            // Mask Update
            FillRectMask.localPosition = selectionBottomLeft;
            FillRectMask.sizeDelta = selectionTopRight - selectionBottomLeft;
        }
        
        public void UpdatePassthroughImagePosition() {
            // Image position calculation so it matches with the passthrough background as close as possible
            m_textureTopLeft = m_cameraController.GetPassthroughTextureTopLeftRay();
            m_textureBottomRight = m_cameraController.GetPassthroughTextureBottomRightRay();

            float distance = m_cameraController.GetCameraDistance();
            Vector3 topLeftWorldPos = m_textureTopLeft.origin + m_textureTopLeft.direction * distance;
            Vector3 bottomRightWorldPos = m_textureBottomRight.origin + m_textureBottomRight.direction * distance;
            Vector3 topLeftScreenPos = m_cameraController.WorldToScreenPoint(topLeftWorldPos);
            Vector3 bottomRightScreenPos = m_cameraController.WorldToScreenPoint(bottomRightWorldPos);

            Vector2 topLeftLocalPos = m_cameraController.ScreenPointToLocalPoint((RectTransform)m_canvas.transform, topLeftScreenPos);
            Vector2 bottomRightLocalPos = m_cameraController.ScreenPointToLocalPoint((RectTransform)m_canvas.transform, bottomRightScreenPos);

            Vector2 bottomLeft = new Vector2(Mathf.Min(topLeftLocalPos.x, bottomRightLocalPos.x), Mathf.Min(topLeftLocalPos.y, bottomRightLocalPos.y));
            Vector2 topRight = new Vector2(Mathf.Max(topLeftLocalPos.x, bottomRightLocalPos.x), Mathf.Max(topLeftLocalPos.y, bottomRightLocalPos.y));
            
            // Readjust Child Image Position to adjust for FillRect Mask Update
            m_fillImageRect.localPosition = bottomLeft - (Vector2)FillRectMask.localPosition;
            m_fillImageRect.sizeDelta = topRight - bottomLeft;
        }
        
        public bool IsSelectionAreaWithinCameraView() {
            if (!FillRectMask || !FillRectMask.gameObject.activeSelf) {
                return false;
            }

            // Get the four corners of the selection rectangle in world space
            Vector3[] corners = new Vector3[4];
            FillRectMask.GetWorldCorners(corners);

            // Check if all corners are within the camera's frustum and viewport
            foreach (Vector3 corner in corners) {
                Vector3 viewportPoint = m_cameraController.WorldToViewportPoint(corner);
                
                // Check if the point is in front of the camera and within viewport bounds
                if (viewportPoint.z <= 0 || viewportPoint.x < 0 || viewportPoint.x > 1 || 
                    viewportPoint.y < 0 || viewportPoint.y > 1) {
                    return false;
                }
            }

            return true;
        }
        
        public RectTransform GetFillRectMask() {
            return FillRectMask;
        }

        public void SetConfig(CameraController cameraController, bool showDebugRect) {
            m_cameraController = cameraController;
            m_showDebugRect = showDebugRect;
        }

        public RectTransform GetLeftCornerBox() {
            return m_leftCornerBox;
        }
        
        public RectTransform GetRightCornerBox() {
            return m_rightCornerBox;
        }

        private static void SetupCornerImage(RectTransform target, bool isLeft, int cornerSpriteSize, int cornerSpriteThickness) {
            var img = target.GetComponent<Image>();
            if (!img) {
                img = target.gameObject.AddComponent<Image>();
            }

            if (!img.sprite) {
                img.sprite = GenerateCornerSprite(isLeft, cornerSpriteSize, cornerSpriteThickness);
            }

            float anchorX = isLeft ? 0 : 1;
            float anchorY = isLeft ? 1 : 0;

            target.anchorMin = target.anchorMax = new Vector2(anchorX, anchorY); // center canvas to camera
            target.pivot = new Vector2(anchorX, anchorY);
            target.sizeDelta = new Vector2(cornerSpriteSize, cornerSpriteSize);
        }

        private static Sprite GenerateCornerSprite(bool isLeft, int cornerSpriteSize, int cornerSpriteThickness) {
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
        
        private static void CleanupCornerSprite(RectTransform cornerBox) {
            if (cornerBox != null) {
                var img = cornerBox.GetComponent<Image>();
                if (img != null && img.sprite != null) {
                    Texture2D texture = img.sprite.texture;
                    Destroy(img.sprite);
                    if (texture != null) {
                        Destroy(texture);
                    }
                }
            }
        }

        public RawImage GetFillImage() {
            return FillImage;
        }
    }
}