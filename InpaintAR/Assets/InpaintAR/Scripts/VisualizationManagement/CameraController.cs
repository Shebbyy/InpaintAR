using Meta.XR;
using UnityEngine;

namespace InpaintAR.Scripts.VisualizationManagement {
    public class CameraController {
        
        private PassthroughCameraAccess m_cameraAccess;
        private Camera m_mainCam;
        
        public CameraController(GameObject obj) {
            InitializeCamera(obj);
        }
        
        private void InitializeCamera(GameObject gameObject) {
            m_cameraAccess = gameObject.AddComponent<PassthroughCameraAccess>();
            m_cameraAccess.CameraPosition = PassthroughCameraAccess.CameraPositionType.Left;
            m_cameraAccess.RequestedResolution = new Vector2Int(1280, 960);
            m_mainCam = Camera.main;
        }

        public Texture GetPassthroughTexture() {
            return m_cameraAccess.GetTexture();
        }

        public Vector2 ScreenPointToLocalPoint(RectTransform transform, Vector2 input) {
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                transform,
                input,
                m_mainCam,
                out Vector2 output
            );

            return output;
        }
        
        public float GetCameraDistance() {
            // use max dimension of canvas to avoid clipping due to rotation, etc. of canvas
            return m_mainCam.farClipPlane - 5f;
        }

        public Transform GetMainCameraTransform() {
            return m_mainCam.transform;
        }

        public Ray GetPassthroughTextureTopLeftRay() {
            return m_cameraAccess.ViewportPointToRay(new Vector2(0, 0));
        }
        public Ray GetPassthroughTextureBottomRightRay() {
            return m_cameraAccess.ViewportPointToRay(new Vector2(1, 1));
        }

        public Vector3 WorldToScreenPoint(Vector3 input) {
            return m_mainCam.WorldToScreenPoint(input);
        }
        
        public Vector3 WorldToViewportPoint(Vector3 input) {
            return m_mainCam.WorldToViewportPoint(input);
        }
    }
}