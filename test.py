import cv2

def recognize_and_track_object(image_path):
    # Load pre-trained YOLOv8 model
    model = cv2.dnn.readNet('yolov8n.onnx')

    # Read image
    image = cv2.imread(image_path)

    # Prepare image for YOLOv8
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    model.setInput(blob)

    # Get model output
    outputs = model.forward(model.getUnconnectedOutLayersNames())

    # Process output
    for output in outputs:
        for detection in output:
            confidence = detection[4]
            if confidence > 0.5:  # Adjust threshold as needed
                class_id = int(detection[5])
                x1, y1, x2, y2 = map(int, detection[0:4])

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label object
                label = f'Class: {class_id}, Confidence: {confidence:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
recognize_and_track_object('image.jpg')
