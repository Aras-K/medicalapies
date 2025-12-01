import requests
import sys
import base64

def test_api():
    url = "http://localhost:8000/api/analyze-xray"
    
    # You need a valid DICOM file path here
    # If you don't have one, the server will return a 400 error about invalid DICOM, 
    # but it confirms the endpoint is reachable.
    file_path = "test_image.dcm" 
    
    print(f"Testing API at {url}")
    
    try:
        # Basic Auth credentials from main.py
        auth = ('client_test', 'TestPass2025!')
        
        # Create a dummy file if it doesn't exist just to hit the endpoint
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                f.write(b'Not a real dicom file')
                
        files = {'file': open(file_path, 'rb')}
        
        response = requests.post(url, files=files, auth=auth)
        
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(response.json())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import os
    test_api()
