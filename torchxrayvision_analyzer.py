# ========================================
# torchxrayvision_analyzer.py - FINAL FIXED VERSION
# Working with proper resize and normalization
# ========================================

import torch
import torchxrayvision as xrv
import numpy as np
import logging
from typing import Dict, Any
import cv2
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class XRayVisionAnalyzer:
    """TorchXRayVision model integration with proper image processing"""
    
    def __init__(self, model_name='densenet121-res224-all'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load pretrained model
            self.model = xrv.models.DenseNet(weights=model_name)
            self.model.to(self.device)
            self.model.eval()
            self.pathologies = self.model.pathologies
            self.model_loaded = True
            logger.info(f"Model loaded: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def preprocess_image(self, pixel_array: np.ndarray) -> torch.Tensor:
        """Preprocess image for model with proper normalization and resizing"""
        
        # Convert to float32
        img = pixel_array.astype(np.float32)
        
        logger.info(f"Original image shape: {img.shape}, range: {img.min():.1f} - {img.max():.1f}")
        
        # Step 1: Normalize to 0-255 range
        img_min = img.min()
        img_max = img.max()
        
        if img_max > img_min:
            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.float32)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        
        logger.info(f"After scaling to 255: {img.min():.1f} - {img.max():.1f}")
        
        # Step 2: Resize to 224x224 (model input size)
        # Using cv2 for resizing
        if img.shape != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized to: {img.shape}")
        
        # Step 3: Apply torchxrayvision normalization
        # This expects values in 0-255 range and normalizes to [-1024, 1024]
        img = xrv.datasets.normalize(img, maxval=255, reshape=True)
        
        # Step 4: Convert to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        
        # Step 5: Add dimensions for batch and channel
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif len(img.shape) == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Final tensor shape: {img.shape}, range: {img.min():.1f} - {img.max():.1f}")
        
        return img.to(self.device)
    
    def analyze(self, pixel_array: np.ndarray) -> Dict[str, Any]:
        """Analyze X-ray image"""
        if not self.model_loaded:
            return {
                "model_available": False,
                "error": "Model not loaded"
            }
        
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(pixel_array)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                predictions = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Process results
            positive_findings = []
            all_probabilities = {}
            
            for i, pathology in enumerate(self.pathologies):
                prob = float(predictions[i])
                all_probabilities[pathology] = round(prob, 3)
                
                # Threshold for positive findings
                if prob > 0.5:
                    positive_findings.append({
                        "pathology": pathology,
                        "probability": round(prob, 3),
                        "confidence": "High" if prob > 0.8 else "Medium" if prob > 0.6 else "Low"
                    })
            
            # Sort by probability
            positive_findings.sort(key=lambda x: x['probability'], reverse=True)
            
            # Generate clinical interpretation
            interpretation = self._generate_interpretation(positive_findings)
            
            logger.info(f"Analysis complete: {len(positive_findings)} positive findings")
            
            return {
                "model_available": True,
                "model_used": "densenet121-res224-all",
                "positive_findings": positive_findings,
                "all_probabilities": all_probabilities,
                "interpretation": interpretation,
                "analysis_status": "success"
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "model_available": True,
                "error": str(e),
                "analysis_status": "failed",
                "troubleshooting": "Check image format and ensure all dependencies are installed"
            }
    
    def _generate_interpretation(self, findings: list) -> Dict[str, Any]:
        """Generate clinical interpretation of findings"""
        
        # Categorize findings by severity
        critical_conditions = ['Pneumothorax', 'Fracture']
        urgent_conditions = ['Pneumonia', 'Edema', 'Effusion', 'Consolidation']
        moderate_conditions = ['Cardiomegaly', 'Atelectasis', 'Mass', 'Nodule']
        
        # Check for critical findings
        critical_findings = [f for f in findings if f['pathology'] in critical_conditions]
        urgent_findings = [f for f in findings if f['pathology'] in urgent_conditions]
        moderate_findings = [f for f in findings if f['pathology'] in moderate_conditions]
        
        # Determine urgency level
        if critical_findings:
            urgency = "🔴 CRITICAL"
            urgency_reason = f"Critical finding detected: {critical_findings[0]['pathology']}"
        elif urgent_findings:
            urgency = "🟠 URGENT"
            urgency_reason = f"Urgent finding detected: {urgent_findings[0]['pathology']}"
        elif moderate_findings:
            urgency = "🟡 MODERATE"
            urgency_reason = f"Abnormality detected: {moderate_findings[0]['pathology']}"
        elif findings:
            urgency = "🟢 LOW"
            urgency_reason = "Minor findings detected"
        else:
            urgency = "🟢 ROUTINE"
            urgency_reason = "No significant abnormalities detected"
        
        # Generate recommendations based on findings
        recommendations = []
        
        # Critical findings recommendations
        if 'Pneumothorax' in [f['pathology'] for f in findings]:
            recommendations.append("⚠️ IMMEDIATE: Consider urgent chest tube placement")
            recommendations.append("Confirm with CT if clinically stable")
        
        if 'Fracture' in [f['pathology'] for f in findings]:
            recommendations.append("⚠️ Evaluate for rib fractures and associated injuries")
        
        # Urgent findings recommendations
        if 'Pneumonia' in [f['pathology'] for f in findings]:
            recommendations.append("Consider antibiotics and supportive care")
            recommendations.append("Follow-up chest X-ray in 4-6 weeks")
        
        if 'Effusion' in [f['pathology'] for f in findings]:
            recommendations.append("Consider thoracentesis if large or symptomatic")
            recommendations.append("Evaluate for underlying cause")
        
        if 'Edema' in [f['pathology'] for f in findings]:
            recommendations.append("Evaluate cardiac function (echo, BNP)")
            recommendations.append("Consider diuretics if clinically indicated")
        
        # Moderate findings recommendations
        if 'Cardiomegaly' in [f['pathology'] for f in findings]:
            recommendations.append("Consider echocardiography for cardiac assessment")
            recommendations.append("Evaluate for heart failure if symptomatic")
        
        if 'Mass' in [f['pathology'] for f in findings] or 'Nodule' in [f['pathology'] for f in findings]:
            recommendations.append("Consider CT for further characterization")
            recommendations.append("Compare with prior imaging if available")
        
        # No findings
        if not findings:
            recommendations.append("✅ No acute cardiopulmonary abnormality")
            recommendations.append("Continue routine care")
        
        # Generate summary
        if findings:
            top_findings = findings[:3]  # Top 3 findings
            summary = "Key findings: " + ", ".join([
                f"{f['pathology']} ({f['probability']*100:.0f}%)" for f in top_findings
            ])
            if len(findings) > 3:
                summary += f" (+{len(findings)-3} more)"
        else:
            summary = "No significant abnormalities detected"
        
        return {
            "urgency_level": urgency,
            "urgency_reason": urgency_reason,
            "summary": summary,
            "recommendations": recommendations,
            "total_findings": len(findings),
            "requires_follow_up": len(findings) > 0,
            "critical_findings_present": len(critical_findings) > 0
        }

# Alternative preprocessing using PIL (backup method)
class AlternativePreprocessor:
    """Backup preprocessing using PIL if cv2 fails"""
    
    @staticmethod
    def preprocess(pixel_array: np.ndarray) -> torch.Tensor:
        # Normalize to 0-255
        img = pixel_array.astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min()) * 255)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img.astype(np.uint8))
        
        # Resize
        img_pil = img_pil.resize((224, 224), Image.LANCZOS)
        
        # Back to numpy
        img = np.array(img_pil).astype(np.float32)
        
        # Apply xrv normalization
        img = xrv.datasets.normalize(img, maxval=255, reshape=True)
        
        # Convert to tensor
        img = torch.from_numpy(img).float()
        
        # Add dimensions
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        
        return img

# Initialize analyzer
_analyzer = None

def analyze_with_ai(pixel_array: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for AI analysis"""
    global _analyzer
    
    try:
        # Initialize analyzer if needed
        if _analyzer is None:
            
            _analyzer = XRayVisionAnalyzer()
        
        # Run analysis
        result = _analyzer.analyze(pixel_array)
        
        # Add metadata context
        result['image_metadata'] = {
            'modality': metadata.get('modality', 'Unknown'),
            'view': metadata.get('view_position', 'Unknown'),
            'body_part': metadata.get('body_part', 'Unknown')
        }
        
        # Add warnings for non-optimal images
        if metadata.get('modality') not in ['CR', 'DX', 'DR', 'RX']:
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append(f"Modality '{metadata.get('modality')}' may not be optimal for chest X-ray analysis")
        
        if metadata.get('view_position') and metadata.get('view_position') not in ['PA', 'AP']:
            result['warnings'] = result.get('warnings', [])
            result['warnings'].append(f"View '{metadata.get('view_position')}' may have reduced accuracy (model trained on PA/AP views)")
        
        logger.info("AI analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        
        # Try alternative preprocessing
        try:
            logger.info("Trying alternative preprocessing...")
            img_tensor = AlternativePreprocessor.preprocess(pixel_array)
            
            if _analyzer and _analyzer.model_loaded:
                with torch.no_grad():
                    outputs = _analyzer.model(img_tensor.to(_analyzer.device))
                    predictions = torch.sigmoid(outputs).cpu().numpy()[0]
                
                # Quick results
                findings = []
                for i, path in enumerate(_analyzer.pathologies):
                    if predictions[i] > 0.5:
                        findings.append(f"{path}: {predictions[i]:.2f}")
                
                return {
                    "model_available": True,
                    "analysis_status": "success (alternative method)",
                    "findings": findings,
                    "note": "Used fallback preprocessing"
                }
        except:
            pass
        
        return {
            "model_available": False,
            "error": str(e),
            "analysis_status": "failed",
        }