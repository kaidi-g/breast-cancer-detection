# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:14:36 2020

@author: Gong Kaidi
"""

#%%
import tensorflow as tf
import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\kaidi\\Desktop\\cancer_detection_homework") 


#%%
#将部分.tif文件转为.jpg文件
from PIL import Image

current_path = os.getcwd()
for root, dirs, files in os.walk(current_path, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print ("A jpeg file already exists for %s" % name)
            # If a jpeg with the name does *NOT* exist, convert one from the tif.
            else:
                outputfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                try:
                    im = Image.open(os.path.join(root, name))
                    print ("Converting jpeg for %s" % name)
                    im.thumbnail(im.size)
                    im.save(outputfile, "JPEG", quality=100)
                except Exception as e: 
                  print(e)


#%%
#读取图片数据

def load_image(image_id, is_train):
    images = np.zeros([300, 300, 3])
    
    if is_train == True:
        path_id = os.path.join(".\\train\\images", image_id)
    else:
        path_id = os.path.join(".\\test\\images", image_id)
        
    for file in [os.path.join(path_id, path_image) for path_image in os.listdir(path_id)]:
        if os.path.splitext(file)[1].lower() != ".tif":
            #print(file)
            image_raw_data = tf.io.gfile.GFile(file, 'rb').read()
            image = tf.io.decode_image(image_raw_data)
            image = tf.image.resize(image, [300, 300])
            #row.append(image)
            images += image
    #images = tf.concat(row, axis=1) if row else None
    return images


#%%
if __name__ == "__main__":
    
    # 1.读取训练集数据
    train_dict = {}
    with open("./train/feats.csv", newline='') as csvfile:
        data = pd.read_csv(csvfile)
        data = data.set_index('id')
        for index, row in data.iterrows():   
            train_dict[index] = {}
            train_dict[index]['X'] = row[['age', 'HER2', 'P53']].values.astype(np.float32)
            train_dict[index]['Y'] = row['molecular_subtype']
            images =  load_image(index, True)
            train_dict[index]['X'] = np.append(train_dict[index]['X'], images)
        
    
    # 2. 构造网络
    model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=[270003, ]),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    

    # 3. 训练
    train_x = np.array([train_dict[key]['X'] for key in train_dict.keys()])
    train_y = np.array([train_dict[key]['Y'] for key in train_dict.keys()]) 
    model.fit(train_x, train_y, epochs=5)
    
    # CNN
    # seq_length = 64
    
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(seq_length, 270003)))
    # model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
    # model.add(tf.keras.layers.MaxPooling1D(3))
    # model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
    # model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
    # model.add(tf.keras.layers.GlobalAveragePooling1D())
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    
    # model.fit(train_x, train_y, batch_size=32, epochs=10)
   
    # 4. 验证：
    test_dict = {}
    with open("./test/feats.csv", newline='') as csvfile:
        test_data = pd.read_csv(csvfile)
        test_data = test_data.set_index('id')
        for index, row in test_data.iterrows():   
            test_dict[index] = {}
            test_dict[index]['X'] = row.values.astype(np.float32)
            images =  load_image(index, False)
            test_dict[index]['X'] = np.append(test_dict[index]['X'], images)
    
    test_x = np.array([test_dict[key]['X'] for key in test_dict.keys()])
    test_y = [np.argmax(one_hot)for one_hot in model.predict(test_x)]
    
    df = pd.DataFrame(      
                         {'id': list(test_dict.keys()),
                          'pred': test_y}
                      )
    df.to_csv(".\\predictions.csv", header = False, index = False)
    


