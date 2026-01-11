using UnityEngine;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;

namespace InpaintAR.Scripts.Input {
    public class AreaSelectionGestureHandler : MonoBehaviour {
        [Header("Hand Skeletons")] 
        [Tooltip("Left Hand of the OVRSkeleton")]
        public OVRSkeleton leftHandSkeleton;

        [Tooltip("Right Hand of the OVRSkeleton")]
        public OVRSkeleton rightHandSkeleton;


        [Header("Gesture Settings")] 
        [Tooltip("Minimum angle for the L-shape (degrees)")] [Range(0f, 180f)]
        public float minAngle = 60f;

        [Tooltip("Maximum angle for the L-shape (degrees)")] [Range(0f, 180f)]
        public float maxAngle = 120f;

        
        [Header("Debug Data")]
        public bool showDebugData = true;

        // Access the detected screen coordinates (for Canvas Overlay)
        public Vector2? LeftHandCornerScreenPos { get; private set; }
        public Vector2? RightHandCornerScreenPos { get; private set; }

        private string m_debugText = "";
        private TextMesh m_debugTextMesh;
        private GameObject m_debugTextObj;
        private Camera m_mainCamera;

        private void Start() {
            m_mainCamera = Camera.main;
        }

        private void Update() {
            // Reset values
            LeftHandCornerScreenPos = null;
            RightHandCornerScreenPos = null;
            
            m_debugText = "Area Detection Debug:\n";

            // check hand data validity
            if (!leftHandSkeleton
                || !leftHandSkeleton.IsDataValid
                || !rightHandSkeleton
                || !rightHandSkeleton.IsDataValid) {
                if (!showDebugData) return;
                
                UpdateDebugDisplay();
                m_debugText += "Waiting for valid hand data...\n";
                if (leftHandSkeleton) m_debugText += $"Left Hand Valid: {leftHandSkeleton.IsDataValid}\n";
                if (rightHandSkeleton) m_debugText += $"Right Hand Valid: {rightHandSkeleton.IsDataValid}\n";

                return;
            }


            // Check Left Hand
            if (CheckLShape(leftHandSkeleton, out Vector3 cornerPosLeft, "Left")) {
                LeftHandCornerScreenPos = m_mainCamera.WorldToScreenPoint(cornerPosLeft);
            }

            // Check Right Hand
            if (CheckLShape(rightHandSkeleton, out Vector3 cornerPosRight, "Right")) {
                RightHandCornerScreenPos = m_mainCamera.WorldToScreenPoint(cornerPosRight);
            }
            if (!showDebugData) return;
            
            UpdateDebugDisplay();
        }

        private void UpdateDebugDisplay() {
            // text output in center about area detection for hand tracking debugging
            if (!m_debugTextObj) {
                m_debugTextObj = new GameObject("AreaDetectionDebug");
                m_debugTextObj.transform.SetParent(m_mainCamera.transform);
                m_debugTextObj.transform.localPosition = new Vector3(0, 0, 2.0f);
                m_debugTextObj.transform.localRotation = Quaternion.identity;
                
                m_debugTextMesh = m_debugTextObj.AddComponent<TextMesh>();
                m_debugTextMesh.characterSize = 0.01f;
                m_debugTextMesh.fontSize = 50;
                m_debugTextMesh.color = Color.green;
                m_debugTextMesh.anchor = TextAnchor.MiddleCenter;
                m_debugTextMesh.alignment = TextAlignment.Center;
            }
            
            m_debugTextObj.SetActive(true);
            if (m_debugTextMesh) m_debugTextMesh.text = m_debugText;
        }

        private bool CheckLShape(OVRSkeleton skeleton, out Vector3 cornerPos, string handName) {
            cornerPos = Vector3.zero;
            var bones = skeleton.Bones;

            // If not enough bones are detected return false, gesture cannot be detected
            if (bones == null || bones.Count < 24) {
                m_debugText += $"{handName}: Not enough bones ({bones?.Count ?? 0})\n";
                return false;
            }

            // Index Finger: Proximal (6) to Distal (8)
            var indexKnuckle = bones[6].Transform.position;
            var indexTip = bones[8].Transform.position;

            // Thumb: Bone 3 (Thumb1) to 5 (Thumb3) more stable than proximal/distal
            var thumbKnuckle = bones[3].Transform.position;
            var thumbTip = bones[5].Transform.position;

            // Finger direction Vectors + normalization
            Vector3 indexVector = (indexTip - indexKnuckle).normalized;
            Vector3 thumbVector = (thumbTip - thumbKnuckle).normalized;

            // Angle between index and thumb
            float angle = Vector3.Angle(indexVector, thumbVector);

            if (showDebugData) {
                Debug.DrawLine(indexKnuckle, indexTip, Color.blue);
                Debug.DrawLine(thumbKnuckle, thumbTip, Color.green);
                m_debugText += $"{handName} Angle: {angle:F1}\n";
            }

            // Slight tolerance to allow for fingers to be slightly off angle
            if (angle >= minAngle && angle <= maxAngle) {
                // Knuckle of thumb is the corner point
                cornerPos = thumbKnuckle;
                m_debugText += $"{handName}: L-Shape Detected!\n";
                return true;
            }

            return false;
        }
    }
}
