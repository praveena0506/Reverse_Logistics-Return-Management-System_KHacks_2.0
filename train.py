import cv2
import pandas as pd
import os
import h5py  # Import h5py for saving to H5 file

# Load CSV metadata
csv_path = "../data/barcode_dataset.csv"  # Update with actual path
df = pd.read_csv(csv_path)


# Function to load and preprocess an image
def load_image(image_path):
    # Try to read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    if image is None:
        print(f"Error: Unable to load image at {image_path}.")
        return None  # Return None if image loading fails

    # Resize image for uniformity
    try:
        image = cv2.resize(image, (200, 200))  # Resize to 200x200
    except cv2.error as e:
        print(f"Error resizing image at {image_path}: {e}")
        return None
    return image


# Create a dictionary to store barcode images
barcode_data = {}

# Iterate over each row in the dataframe to load and process images
for _, row in df.iterrows():
    product_id = row["product_id"]
    image_path = row["barcode"]

    # Check if the image path exists before processing
    if os.path.exists(image_path):
        image = load_image(image_path)
        if image is not None:  # Only add the image if it was successfully loaded
            barcode_data[product_id] = image
    else:
        print(f"Error: Image path does not exist: {image_path}")

# Save the barcode data to an H5 file
with h5py.File("barcode_model.h5", "w") as f:
    for product_id, image in barcode_data.items():
        # Save each barcode image under its product_id in the H5 file
        f.create_dataset(str(product_id), data=image)

print("Model training complete. Barcode data saved as 'barcode_model.h5'.")
