import cv2
import numpy as np
import onnxruntime as ort

class YOLOv8ONNX:
    def __init__(self, onnx_model_path: str, classes: list):
        self.model_path = onnx_model_path
        self.classes = classes
        self.session = None
        self.input_name = None
        
        try:
            # Try loading the model if it exists
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            inputs = self.session.get_inputs()
            if inputs and len(inputs) > 0:
                self.input_name = inputs[0].name
            print(f"[*] ONNX Model loaded from: {self.model_path}")
        except Exception as e:
            print(f"[!] Warning: Could not load ONNX model. Predict will return mock data. ({e})")
            
    def preprocess(self, image: np.ndarray):
        # OpenCV loads in BGR, YOLOv8 expects RGB
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize, normalize and transpose image to fit YOLOv8 requirements
        input_img = cv2.resize(input_img, (640, 640))
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0) # Add batch dimension
        input_img = input_img.astype('float32') / 255.0 # Normalize 0-1
        return input_img

    def predict(self, image: np.ndarray, conf_threshold=0.5):
        if self.session is None or self.input_name is None:
            # Mock return for demonstration when model file is missing
            return [{"class": self.classes[0], "confidence": 0.95, "bbox": [100, 100, 300, 300]}]

        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Output shape is typically [1, 4 + num_classes, 8400]
        output = outputs[0]
        
        # Squeeze batch dimension: [4 + num_classes, 8400]
        output = np.squeeze(output)
        
        # Transpose to get [8400, 4 + num_classes] for easier iteration
        output = output.transpose()
        
        results = []
        
        # Iterate through predictions
        for row in output:
            # First 4 are bounding box coords (xc, yc, w, h)
            # Rest are class scores
            class_scores = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(class_scores)
            class_id = max_idx[1]
            confidence = class_scores[class_id]
            
            if confidence >= conf_threshold:
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                
                # Convert center to min max coords
                x_min = xc - w / 2
                y_min = yc - h / 2
                
                # YOLOv8 ONNX output is relative to the internal 640x640 size.
                # In a full app, scale these back to the original image dimensions.
                # We normalize them relative to 640x640 input for the API format.
                results.append({
                    "class": self.classes[class_id] if class_id < len(self.classes) else "Unknown",
                    "confidence": float(confidence),
                    "bbox": [float(x_min), float(y_min), float(w), float(h)]
                })
        
        # Note: A true production implementation would apply Non-Maximum Suppression (NMS) here.
        # But for this demo, we'll sort by confidence and return the highest confident box.
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return [results[0]] if len(results) > 0 else []
        
        return results
