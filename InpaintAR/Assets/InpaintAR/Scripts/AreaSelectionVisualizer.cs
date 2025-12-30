using UnityEngine;
using UnityEngine.UI;

namespace InpaintAR.Scripts {
    public class AreaSelectionVisualizer : MonoBehaviour {
        [Header("RequiredData")] 
        [Tooltip("Area Detection Object. If null, attempts to find it in the scene.")]
        public AreaDetection areaDetection;

        [Tooltip("Size of the corner sprites image")]
        public int cornerSpriteSize = 128;
        [Tooltip("Thickness of the corner sprite lines")]
        public int cornerSpriteThickness = 50;
        
        public RectTransform FillRect { get; private set; }

        // Internal references to the generated UI elements
        private RectTransform m_leftCornerBox;
        private RectTransform m_rightCornerBox;
        private Canvas m_canvas;
        
        private Image FillArea { get; set; }
        
        [Header("Debug Data")]
        public bool showDebugRect = true;

        private void Start() {
            if (!areaDetection) {
               Debug.LogError("AreaSelectionVisualizer: AreaDetection not set!");
               return;
            }

            CreateUICanvasAndCorners();
        }

        void Update() {
            if (   !areaDetection 
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
                && areaDetection.LeftHandCornerScreenPos.Value.y >= areaDetection.RightHandCornerScreenPos.Value.y)  {
                FillRect.gameObject.SetActive(true);
                
                var leftPos = m_leftCornerBox.localPosition;
                var rightPos = m_rightCornerBox.localPosition;
                
                Vector2 bottomLeft = new Vector2(Mathf.Min(leftPos.x, rightPos.x), Mathf.Min(leftPos.y, rightPos.y));
                Vector2 topRight = new Vector2(Mathf.Max(leftPos.x, rightPos.x), Mathf.Max(leftPos.y, rightPos.y));

                FillRect.localPosition = bottomLeft;
                FillRect.sizeDelta = topRight - bottomLeft;
            }
            else {
                FillRect.gameObject.SetActive(false);
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
                m_canvas.worldCamera,
                out Vector2 localPos
            );

            cornerElem.localPosition = localPos;
        }
        
        private void CreateUICanvasAndCorners() {
            GameObject canvasObj = new GameObject("AreaSelectionUICanvas");
            m_canvas = canvasObj.AddComponent<Canvas>();
            // set in front of camera, due to ui being rendered separately from world -> plane distance fine
            m_canvas.renderMode = RenderMode.ScreenSpaceCamera;
            m_canvas.worldCamera = Camera.main;
            m_canvas.planeDistance = Camera.main.farClipPlane - 0.1f;
            
            CanvasScaler scaler = canvasObj.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);

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
            
            // Create the fill area game object
            GameObject fillObj = new GameObject("FillArea");
            fillObj.transform.SetParent(canvasObj.transform, false);
            FillRect = fillObj.AddComponent<RectTransform>();
            
            FillRect.pivot = new Vector2(0, 0);
            FillRect.anchorMin = FillRect.anchorMax = new Vector2(0.5f, 0.5f);

            // Add an Image to fill it with red initially
            FillArea = fillObj.AddComponent<Image>();
            if (showDebugRect) {
                FillArea.color = Color.red;
            }
            else {
                FillArea.color = Color.clear;
            }

            // Start hidden
            FillRect.gameObject.SetActive(false);
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
    }
}
