from __future__ import print_function
import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
DATASET_PATH = 'images/' # the dataset file or root folder path.

# Image Parameters
N_CLASSES = 5 # CHANGE HERE, total number of classes
IMG_HEIGHT = 500 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 500 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale

def get_square(image,square_size):

    height,width=image.shape
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4

    mask = np.zeros((differ,differ), dtype="uint8")   
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    mask=cv.resize(mask,(square_size,square_size),interpolation=cv.INTER_AREA)

    return mask 

# Reading the dataset
def read_images(dataset_path, batch_size):
    imagepaths, labels = list(), list()
    # An ID will be affected to each sub-folders by alphabetical order
    label = 0
    # List the directory
    try:  # Python 2
        classes = sorted(os.walk(dataset_path).next()[1])
    except Exception:  # Python 3
        classes = sorted(os.walk(dataset_path).__next__()[1])
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        try:  # Python 2
            walk = os.walk(c_dir).next()
        except Exception:  # Python 3
            walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps jpg images
            if sample.endswith('.jpg'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1
    

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    #image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    image = get_square(image,500)

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, y

def main():

    
    batch_size = 1000
    # Build the data input
    X, y = read_images(DATASET_PATH, batch_size)

    print ("X: ", X)
    print ("y: ", y)




if __name__=='__main__':
    main()
