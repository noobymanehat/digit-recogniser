
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageDraw
import tkinter as tk

print(" Loading MNIST dataset")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(" Building model")
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("🔹 Training model (3 epochs)...")
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nModel ready! Test Accuracy: {test_acc:.4f}")

window = tk.Tk()
window.title("🖊️ Draw a Digit (0–9)")

canvas_size = 280
canvas = tk.Canvas(window, width=canvas_size, height=canvas_size, bg='white')
canvas.pack()


image1 = Image.new("L", (canvas_size, canvas_size), "white")
draw = ImageDraw.Draw(image1)

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
    draw.ellipse([x1, y1, x2, y2], fill='black')

canvas.bind("<B1-Motion>", paint)

def predict_digit():
    img_resized = image1.resize((28, 28))
    img_array = np.array(img_resized)
    img_array = 255 - img_array  
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    prediction = np.argmax(model.predict(img_array))
    result_label.config(text=f"Predicted Digit: {prediction}")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill="white")
    result_label.config(text="")

tk.Button(window, text="Predict", command=predict_digit, width=20, bg="lightgreen").pack(pady=10)
tk.Button(window, text="Clear", command=clear_canvas, width=20, bg="lightcoral").pack()
result_label = tk.Label(window, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

window.mainloop()
