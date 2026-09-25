import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === 1. LOAD & SỬA DỮ LIỆU ===
dataset = loadtxt('data_CKD_07IN_27_08_2024_001_AI.csv', delimiter=',')

# LỌC DỮ LIỆU: Chỉ giữ y trong [0, 1]
valid = (dataset[:, 9] >= 0) & (dataset[:, 9] <= 1) & (~np.isnan(dataset[:, 9]))
dataset = dataset[valid]
print(f"Dữ liệu hợp lệ: {len(dataset)} mẫu")

# LẤY CHÍNH XÁC 8 CỘT ĐẦU (Index 0 đến 7)
x = dataset[:, 0:8]
y = dataset[:, 8]

# === 2. CHIA DỮ LIỆU ===
x_train_val, x_test, y_train_val, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val, y_train_val, test_size=0.2, random_state=42
)

# === 3. CHUẨN HÓA ===
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_val = scaler.transform(x_val)
X_test = scaler.transform(x_test)


# === 4. MÔ HÌNH (input_dim=8) ===
# model = Sequential()
# model.add(Dense(256, input_dim=8, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()

model = load_model("mymodel_CKD_07IN_DOI_27_08_2024_001_AI.h5", compile=False)

# DÙNG MSE/MAE (Regression)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# === 5. TRAIN ===
# model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# === 6. LƯU ===
# model.save("mymodel_CKD_07IN_DOI_27_08_2024_001_AI.h5")

# === 7. ĐÁNH GIÁ ===
loss, mae = model.evaluate(X_test, y_test, verbose=0) 
print("Loss = ", loss)
print("MAE = ", mae)

# === 8. DỰ ĐOÁN MẪU MỚI ===
x_new_raw = [180, 75, 6.6, 26, 24.1, 9, 250, 45]
y_new = [0.5]

x_new_array = np.array([x_new_raw])
X_new_scaled = scaler.transform(x_new_array) 

y_predict = model.predict(X_new_scaled)

# === 9. PHẦN PRINT ===
print(f"-x_new={x_new_raw}---------------------------------------------")
print("----------------------------------------------------------------------------------------")
print(" PROJECT IN ISEF  - DETECT CKD USING AI")
print("TRUE = ", y_new)
print("AI PREDICT = ", y_predict)
print("dự án AI PREDICT = ", y_predict)
print("----------------------------------------------------------------------------------------")