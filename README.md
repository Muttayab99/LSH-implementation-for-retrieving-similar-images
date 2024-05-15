**Similar Images Retrieval using Locality-Sensitive Hashing (LSH)**


This repository contains a Flask-based web application for retrieving similar images from a dataset using Locality-Sensitive Hashing (LSH). The application allows users to upload an image and find visually similar images from a predefined dataset.


To run this project, you need Python 3 and the following libraries:

opencv-python

numpy

scikit-learn

imutils

Install the required libraries using pip:

bash

Copy code

opencv-python numpy scikit-learn imutils

Usage

**Prepare the dataset:** 

Ensure your image dataset is located in the specified folder path. Modify the folder_path variable in the code to point to your dataset.

Project Structure

Loading Image Paths

The image paths from the dataset are loaded and stored in a list:

python

Copy code

folder_path = "C:\\Users\\Muttayab\\Desktop\1\\Dataset"

image_paths = list(paths.list_images(folder_path))

Computing Histograms

Histograms are computed for each image in the dataset to represent their color distribution:

python

Copy code

def compute_histogram(image):

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist


dataset_histograms = [compute_histogram(cv2.imread(img_path)) for img_path in image_paths]

dataset_histograms = np.array(dataset_histograms)

Locality-Sensitive Hashing (LSH)

LSH is implemented to hash the histograms into hash tables:


python

Copy code

num_tables = 5

hash_size = 10

hash_tables = [[] for _ in range(num_tables)]

planes = [np.random.randn(len(uploaded_hist)) for _ in range(num_tables * hash_size)]


for i, hist in enumerate(dataset_histograms):

    for j in range(num_tables):
    
        hash_val = "".join(['1' if np.dot(hist, plane) > 0 else '0' for plane in planes[j * hash_size: (j + 1) * hash_size]])
        
        hash_tables[j].append((hash_val, i))
        
Query Execution

The uploaded image is processed, hashed, and compared to the images in the hash tables to find similar images using cosine similarity:


python

Copy code

uploaded_hist = compute_histogram(uploaded_img)

similar_images = []

for j in range(num_tables):

    query_hash_val = "".join(['1' if np.dot(uploaded_hist, plane) > 0 else '0' for plane in planes[j * hash_size: (j + 1) * hash_size]])
    
    similar_found = [index for hash_val, index in hash_tables[j] if hash_val == query_hash_val]

    candidate_histograms = dataset_histograms[similar_found]
    
    similarities = cosine_similarity(uploaded_hist.reshape(1, -1), candidate_histograms)

    top_similar_indices = np.argsort(similarities)[-5:][::-1]
    
    top_similar_images = [image_paths[index] for index in similar_found[top_similar_indices]]
    
    similar_images.extend(top_similar_images)
