import os
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
# Ensure OPENAI_API_KEY is set in environment variables
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    logger.warning(f"OpenAI client could not be initialized: {e}")
    client = None

def generate_medical_report(analysis_results: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """
    Generate a natural language medical report based on AI analysis findings.
    
    Args:
        analysis_results: Dictionary containing AI findings, probabilities, etc.
        metadata: Dictionary containing DICOM metadata (patient_id, etc.)
        
    Returns:
        str: Generated medical report or error message
    """
    if not client:
        return "Error: OpenAI API key not configured. Cannot generate report."
        
    try:
        # Construct the prompt
        findings_text = ""
        if analysis_results.get("positive_findings"):
            findings_text = "Positive Findings:\n"
            for finding in analysis_results["positive_findings"]:
                findings_text += f"- {finding['pathology']}: {finding['probability']*100:.1f}% confidence ({finding['confidence']})\n"
        else:
            findings_text = "No significant positive findings detected.\n"
            
        interpretation = analysis_results.get("interpretation", {})
        urgency = interpretation.get("urgency_level", "Unknown")
        summary = interpretation.get("summary", "No summary available")
        recommendations = "\n".join(interpretation.get("recommendations", []))
        
        patient_info = f"Patient ID: {metadata.get('patient_id', 'Unknown')}, Body Part: {metadata.get('body_part', 'Unknown')}"
        
        prompt = f"""
        You are an expert radiologist assistant. Write a concise, professional medical report for a Chest X-Ray based on the following AI analysis data.
        
        CONTEXT:
        {patient_info}
        
        AI ANALYSIS DATA:
        Urgency Level: {urgency}
        AI Summary: {summary}
        
        {findings_text}
        
        AI Recommendations:
        {recommendations}
        
        INSTRUCTIONS:
        1. Write a structured report with sections: "Clinical Indication" (if available, otherwise say 'Not provided'), "Findings", and "Impression".
        2. In "Findings", describe the observations based on the AI data. Be objective.
        3. In "Impression", summarize the key diagnostic conclusions and urgency.
        4. Include the recommendations in the Impression or a separate "Recommendations" section.
        5. Use professional medical terminology.
        6. If no abnormalities are found, state "Normal chest X-ray" clearly.
        7. Add a disclaimer at the end: "This report is AI-generated and must be verified by a qualified radiologist."
        """
        
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo if preferred for cost
            messages=[
                {"role": "system", "content": "You are a helpful and precise medical AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Low temperature for more deterministic/factual output
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating LLM report: {e}")
        return f"Error generating report: {str(e)}"
