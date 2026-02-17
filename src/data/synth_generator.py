import numpy as np
import tensorflow as tf

def generate_synthetic_xray(image_size=128, fracture_type=0, hospital_id=0):
    """
    Generate a synthetic femur-like X-ray image.

    fracture_type: 0=normal, 1=atypical
    hospital_id: used to simulate domain shift
    """

    img = np.zeros((image_size, image_size))

    # femur shaft
    cv = image_size // 2
    img[:, cv-5:cv+5] = 0.7

    # fracture pattern
    if fracture_type == 1:
        img[image_size//2: image_size//2+3, :] = 0.9

    # hospital-specific brightness shift
    img += hospital_id * 0.01

    img = np.clip(img, 0, 1)
    return img[..., None]


def create_dataset(n_samples=1000, n_hospitals=10):
    images, labels, hospitals = [], [], []

    for i in range(n_samples):
        fracture = np.random.randint(0, 2)
        hospital = np.random.randint(0, n_hospitals)

        img = generate_synthetic_xray(fracture_type=fracture,
                                       hospital_id=hospital)

        images.append(img)
        labels.append(fracture)
        hospitals.append(hospital)

    return np.array(images), np.array(labels), np.array(hospitals)
