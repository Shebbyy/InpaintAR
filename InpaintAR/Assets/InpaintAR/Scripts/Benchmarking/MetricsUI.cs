using InpaintAR.Scripts.Benchmarking.Evaluators;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace InpaintAR.Scripts.Benchmarking {
    public class MetricsUI : MonoBehaviour {
        private TextMesh m_textMesh;
        private GameObject m_uiObj;
        private GameObject m_textObj;

        void Start() {
            m_uiObj = new GameObject("MetricsUI");
            m_uiObj.transform.SetParent(Camera.main?.transform);
            m_uiObj.transform.localPosition = new Vector3(0, 0, 2.0f);
            m_uiObj.transform.localRotation = Quaternion.identity;
                
            m_textMesh = m_uiObj.AddComponent<TextMesh>();
            m_textMesh.characterSize = 0.01f;
            m_textMesh.fontSize = 50;
            m_textMesh.color = Color.green;
            m_textMesh.anchor = TextAnchor.UpperLeft;
            m_textMesh.alignment = TextAlignment.Left;
        }

        void Update() {
            UpdateMetricUI();
        }

        private void UpdateMetricUI() {
            string text = $"Current FPS: {PerformanceEvaluator.GetTotalFPS():F2}\n";
            text += $"Inpainting Only FPS: {PerformanceEvaluator.GetInpaintingIsolatedFPS():F2}\n";
            text += $"Average Inpainting Time (ms): {PerformanceEvaluator.GetAverageInpaintingTime():F2}\n";
            text += $"Average TPP (TimePerPixel in ms): {PerformanceEvaluator.GetAverageTimePerPixel():F2}\n\n";
            
            text += $"Average Inpainting Quality: {QualityEvaluator.GetAverageQuality():F2}\n";
            if (m_textMesh) m_textMesh.text = text;
        }
    }
}