import sys
import os

# Add the project directory to the path so we can import the modules
sys.path.append("/Users/aras.koplanto/Documents/medical API SERVICE/Xray_api")

from llm_service import generate_medical_report

def test_llm_generation():
    print("Testing LLM Report Generation...")
    
    # Mock data
    mock_analysis_results = {
        "positive_findings": [
            {"pathology": "Pneumonia", "probability": 0.85, "confidence": "High"},
            {"pathology": "Effusion", "probability": 0.65, "confidence": "Medium"}
        ],
        "interpretation": {
            "urgency_level": "🟠 URGENT",
            "summary": "Key findings: Pneumonia (85%), Effusion (65%)",
            "recommendations": [
                "Consider antibiotics",
                "Follow-up X-ray in 4-6 weeks"
            ]
        }
    }
    
    mock_metadata = {
        "patient_id": "TEST_PATIENT_001",
        "body_part": "CHEST"
    }
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Test will likely fail or return error message.")
    
    # Run generation
    report = generate_medical_report(mock_analysis_results, mock_metadata)
    
    print("\nGenerated Report:")
    print("-" * 40)
    print(report)
    print("-" * 40)
    
    if "Error" in report and "API key" in report:
        print("Test Result: PASSED (Graceful failure without API key)")
    elif "Clinical Indication" in report or "Findings" in report:
        print("Test Result: PASSED (Report generated)")
    else:
        print("Test Result: UNCERTAIN (Check output)")

if __name__ == "__main__":
    test_llm_generation()
