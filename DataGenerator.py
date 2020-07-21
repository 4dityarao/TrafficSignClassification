from keras.preprocessing.image import ImageDataGenerator, img_to_array
from numpy import expand_dims
import os
import cv2

PATH = "C:\\Users\Aditya\\Desktop\\TEMProject\\Dataset"
PATH2= "C:\\Users\Aditya\\Desktop\\TEMProject\\Validation Set"
WIDTH = 81
HEIGHT = 81
def create_training_images():
    for i in range(0,43):
        str = PATH + "{}{}".format("\\", i)


        if not os.path.exists(str):
            print("Skipped ",i)
            continue

            for file in os.listdir(str):
                if file == '{}.jpg'.format(i):

                    img = cv2.imread(os.path.join(str, file))
                    data = img_to_array(img)
                    samples = expand_dims(data, 0)

                    datagen = ImageDataGenerator(rotation_range=18,
                                                 width_shift_range=0.15,
                                                 height_shift_range=0.15,
                                                 shear_range=0.08,
                                                 zoom_range=[1.0,2.0], brightness_range=[0.7, 2], rescale=1/255, fill_mode='nearest')

                    # it = datagen.flow(samples, batch_size=1,save_to_dir=PATH,save_format='jpg')

                    for y, m in zip(datagen.flow(samples, save_to_dir=str, save_prefix='manmade', save_format='jpg'),
                                    range(500)):

                        pass
                break



def create_validation_images():
    for i in range(43):
        str = PATH2 + "{}{}".format("\\", i)

        if not os.path.exists(str):
            print("Skipped ", i)
            continue
        else:
            print(os.listdir(str))

            for file in os.listdir(str):
                img = cv2.imread(os.path.join(str, file))
                data = img_to_array(img)
                samples = expand_dims(data, 0)
                # create image data augmentation generator
                datagen = ImageDataGenerator(rotation_range=15,
                                             width_shift_range=0.10,
                                             height_shift_range=0.10,
                                             shear_range=0.1,
                                             zoom_range=[1.0,2.0], brightness_range=[.6, 5],rescale=1/255, fill_mode='nearest')

                # it = datagen.flow(samples, batch_size=1,save_to_dir=PATH,save_format='jpg')

                for y, m in zip(datagen.flow(samples, save_to_dir=str, save_prefix='manmade', save_format='jpg'),
                                range(250)):
                    pass
                break




#create_training_images()

create_validation_images()