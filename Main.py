from flask import Flask, render_template, request
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from imutils import paths

app = Flask(__name__)

# Loading image paths
folder_path = "C:\\Users\\Muttayab\\Desktop\\Assignment 1\\Dataset"
image_paths = list(paths.list_images(folder_path))

# Function for computing histogram
def compute_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Computing histograms for images in dataset
dataset_histograms = []
for img_path in image_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = compute_histogram(img)
    dataset_histograms.append(hist)

# Converting to numpy array
dataset_histograms = np.array(dataset_histograms)

# Displaying the home page
@app.route('/')
def home():
    return render_template('index.html')

# check for presence of image
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Reading and pre-processing the uploaded image
    uploaded_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    uploaded_img = cv2.cvtColor(uploaded_img, cv2.COLOR_BGR2RGB)
    uploaded_hist = compute_histogram(uploaded_img)

    # Implementing LSH
    num_tables = 5
    hash_size = 10
    hash_tables = [[] for _ in range(num_tables)]
    planes = [np.random.randn(len(uploaded_hist)) for _ in range(num_tables * hash_size)]

    for i, hist in enumerate(dataset_histograms):
        for j in range(num_tables):
            hash_val = ""
            for plane in planes[j * hash_size: (j + 1) * hash_size]:
                hash_val += '1' if np.dot(hist, plane) > 0 else '0'
            hash_tables[j].append((hash_val, i))

    # Execute queries against hash tables
    similar_images = []

    for j in range(num_tables):
        query_hash_val = ""
        for plane in planes[j * hash_size: (j + 1) * hash_size]:
            query_hash_val += '1' if np.dot(uploaded_hist, plane) > 0 else '0'

        # Retrieve candidate matches from hash table
        similar_found = [index for hash_val, index in hash_tables[j] if hash_val == query_hash_val]

        # Analyze similarity using cosine similarity
        candidate_histograms = dataset_histograms[similar_found]
        similarities = cosine_similarity(uploaded_hist.reshape(1, -1), candidate_histograms)

        # Select the top similar images
        top_similar_indices = np.argsort(similarities)[-5:][::-1]
        top_similar_images = [image_paths[index] for index in similar_found[top_similar_indices]]
        similar_images.extend(top_similar_images)

    return render_template('index.html', uploaded_image=uploaded_img, similar_images=similar_images)

if __name__ == '__main__':
    app.run(debug=True)
