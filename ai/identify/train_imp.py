import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output as cls



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import pydot_ng as pydot
from tensorflow.keras.utils import plot_model

# Thiết lập thông số
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_SIZE = ( IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 10  # Số lượng mẫu xử lý trong cùng 1 lúc -> 1 mẫu
EPOCHS = 200 # Số lần lặp của toàn bộ dữ liệu huấn luyện

# Đường dẫn
data_csv_path = 'A:/LV/ai/identify/dataset/dataset.csv'
data_img_dir = 'A:/LV/ai/identify/dataset/seafood'


# Đọc file CSV 
data_csv = pd.read_csv(data_csv_path)


# # Hiển thị
# print(data_csv.head())

# Lấy đường dẫn ảnh
data_csv['filename'] = [data_img_dir + f"/{filename}" for filename in data_csv['filename']]

# # Hiển thị
# print(data_csv.head())


labels = [str(word).strip() for word in data_csv['name'].to_numpy()]

# Tìm các nhãn duy nhất và sắp xếp để đảm bảo thứ tự nhất quán
unique_labels = sorted(set(labels))

# Chuyển đổi nhãn thành các giá trị số
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Số lượng lớp
num_classes = len(unique_labels)

print(f"Số lượng lớp: {num_classes}")
print(f"Giá trị nhãn lớn nhất: {max(encoded_labels)}")

# def extract_features(image):
#     """
#     Trích xuất nhiều đặc trưng khác nhau từ ảnh đầu vào
#     """
#     # Chuyển về ảnh grayscale và thêm batch dimension
#     gray_image = tf.image.rgb_to_grayscale(image)
#     gray_image = tf.expand_dims(gray_image, 0)  # Thêm batch dimension
    
#     # 1. Sobel Edge Detection
#     sobel_edges = tf.image.sobel_edges(gray_image)
#     sobel_x = sobel_edges[0,:,:,:,0]  # Lấy kết quả theo trục x
#     sobel_y = sobel_edges[0,:,:,:,1]  # Lấy kết quả theo trục y
#     edge_magnitude = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))
    
#     # 2. Local Binary Pattern (LBP)-like texture features
#     def get_local_patterns(img):
#         img = tf.squeeze(img)  # Loại bỏ batch dimension
#         # Tạo các phiên bản dịch chuyển của ảnh
#         shifts = [
#             tf.roll(img, shift=1, axis=1),  # right
#             tf.roll(img, shift=-1, axis=1),  # left
#             tf.roll(img, shift=1, axis=0),  # down
#             tf.roll(img, shift=-1, axis=0),  # up
#         ]
#         # So sánh với pixel trung tâm
#         patterns = tf.stack([shift > img for shift in shifts], axis=-1)
#         return tf.cast(patterns, tf.float32)
    
#     texture_patterns = get_local_patterns(gray_image)
    
#     # 3. Color features
#     hsv_image = tf.image.rgb_to_hsv(image)
    
#     # Kết hợp tất cả các đặc trưng
#     features = tf.concat([
#         image,                    # RGB channels (3)
#         edge_magnitude,           # Edge features (1)
#         texture_patterns,         # Texture features (4)
#         hsv_image,               # HSV color space (3)
#     ], axis=-1)
    
#     return features
def load_and_preprocess_image(img_path: str, augment: bool = False):
    # Đọc ảnh
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Thay đổi kích thước
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    
    # Chuẩn hóa
    image = tf.cast(image, tf.float32) / 255.0
    
    if augment:
        # Các augmentation hiện tại của bạn
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        
        # Thêm các augmentation mới
       
        # Zoom ngẫu nhiên
        image = tf.image.random_crop(
            tf.image.resize(image, [int(IMG_HEIGHT*1.1), int(IMG_WIDTH*1.1)]),
            [IMG_HEIGHT, IMG_WIDTH, 3]
        )
        
        # Thêm noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
        image = image + noise
        
        # Đảm bảo các giá trị hình ảnh vẫn ở [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
    # # Trích xuất đặc trưng
    # features = extract_features(image)
    
    return image



# # Tạo dữ liệu để chuẩn bị cho việc huấn luyện
def create_dataset(file_paths, labels, batch_size, augment=False):
    def map_func(x, label):
        return load_and_preprocess_image(x, augment), label

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset





# Tạo dataset
train_file_paths = data_csv['filename']
train_labels = encoded_labels
train_ds = create_dataset(train_file_paths, train_labels, BATCH_SIZE, augment=True)

print(f"Training Data Size   : {tf.data.Dataset.cardinality(train_ds).numpy() * BATCH_SIZE}")



# Hiển thị ảnh
def show_images(dataset, num_images=4, figsize=(25, 25), model=None, label_encoder=None):
    plt.figure(figsize=figsize)
    
    for i, (image, label) in enumerate(dataset.take(num_images)):
        ax = plt.subplot(2, 2, i + 1)
        
        plt.imshow(image[0].numpy())
        
        # Giải mã nhãn thực từ chỉ số thành tên
        true_name_idx = label[0].numpy()  # Lấy chỉ số nhãn từ tensor
        true_name = label_encoder.inverse_transform([true_name_idx])[0] if label_encoder else str(true_name_idx)
        
        true_label = f"Name: {true_name}\n"
      
        # Nếu có mô hình, thực hiện dự đoán và hiển thị kết quả
        if model:
            predictions = model.predict(tf.expand_dims(image[0], axis=0))  # Thêm chiều để phù hợp với batch đầu vào
            pred_name_idx = np.argmax(predictions[0])
            
            # Giải mã nhãn dự đoán thành tên
            pred_name = label_encoder.inverse_transform([pred_name_idx])[0] if label_encoder else str(pred_name_idx)
            
            pred_label = f"Pred Name: {pred_name}\n"
    
            plt.title(f"{true_label}\n{pred_label}", fontsize=10)
        else:
            plt.title(true_label, fontsize=10)
        
        plt.axis('off')
    
    plt.subplots_adjust(top=0.9, bottom=0.1) 
    plt.show()

    
# Hiển thị ảnh
# show_images(train_ds, label_encoder=label_encoder)


input_layer = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name="image")

# Convolutional layers
x = layers.Conv2D(
    filters=32, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal'
)(input_layer)

x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = layers.Conv2D(
    filters=32, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal'
)(x)

x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = layers.Conv2D(
    filters=64, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal'
)(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

x = layers.Conv2D(
    filters=64, 
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    kernel_initializer='he_normal'
)(x)
x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
x = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x)
x = layers.Dropout(0.2)(x)

name_output = layers.Dense(num_classes, activation='softmax')(x)

# Create model
model = keras.Model(inputs=input_layer, outputs=name_output)

model.summary() 

# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# history = model.fit(
#     train_ds,
#     epochs=EPOCHS,
#     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
# )

loaded_model = keras.models.load_model('model_seafood.keras')

# model_save_path = 'A:/LV/ai/identify/seafood_model_improve.keras'  
# model.save(model_save_path)
plot_model(loaded_model, to_file='model.png', show_shapes=True, show_layer_names=True)

# def load_single_image(img_path: str):
#     # Tải và xử lý hình ảnh đơn lẻ
#     image = tf.io.read_file(img_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [250, 250])  # Đặt lại kích thước ảnh về 200x200
#     image = tf.cast(image, tf.float32) / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
#     image = tf.expand_dims(image, axis=0)  # Thêm chiều batch
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_brightness(image, max_delta=0.2)
#     image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
#     image = tf.image.random_hue(image, max_delta=0.2)
#     image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
#     return image


# def predict_single_image(model, img_path, label_encoder):
#     # Tải hình ảnh
#     image = load_single_image(img_path)

#     # Dự đoán
#     predictions = model.predict(image)
    
#     # Lấy chỉ số dự đoán và giải mã thành nhãn
#     predicted_idx = np.argmax(predictions[0])
#     predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

#     return predicted_label


# img_path = 'A:/LV/ai/identify/test/images/2599.png'  
# # print(load_single_image(img_path))
# predicted_label = predict_single_image(loaded_model, img_path, label_encoder)
# print(predicted_label)
