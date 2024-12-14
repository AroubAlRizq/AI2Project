<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Processing</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #2e0357;
            color: white;
            text-align: center;
        }
        h1 {
            margin-bottom: 20px;
        }
        .container {
            max-width: 600px;
            margin: auto;
            background-color: #50157e;
            padding: 20px;
            border-radius: 10px;
        }
        img {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <h1>AI Image Processing</h1>
    <div class="container">
        <h2>Upload an Image</h2>
        <input type="file" id="image-upload" accept="image/*">
        <button onclick="processImage()">Process Image</button>

        <h3>Classification Result</h3>
        <p id="classification-result">None</p>

        <h3>Segmentation Result</h3>
        <img id="segmentation-result" src="" alt="Segmentation Output" style="display: none;">

        <h3>Detection Result</h3>
        <img id="detection-result" src="" alt="Detection Output" style="display: none;">
    </div>

    <script>
        function processImage() {
            const imageUpload = document.getElementById('image-upload');
            if (!imageUpload.files[0]) {
                alert('Please upload an image!');
                return;
            }

            const formData = new FormData();
            formData.append('file', imageUpload.files[0]);

            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('classification-result').textContent = 
                    data.classification === 1 ? 'Class 1' : 'Class 0';

                document.getElementById('segmentation-result').src = data.segmentation;
                document.getElementById('segmentation-result').style.display = 'block';

                document.getElementById('detection-result').src = data.detection;
                document.getElementById('detection-result').style.display = 'block';
            })
            .catch(err => alert('Error: ' + err.message));
        }
    </script>
</body>
</html>
