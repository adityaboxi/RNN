import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense

print("1. Initializing Sequential RNN Model...")
model = Sequential([
    Input(shape=(4, 5)),
    SimpleRNN(3),
    Dense(1, activation='sigmoid')
])

# Print the model architecture summary
model.summary()

print("\n2. Creating Dummy Dataset...")
X_dummy = np.random.random((1, 4, 5)) 
print(f"Shape of dummy input data: {X_dummy.shape}")

print("\n3. Running Forward Propagation...")
prediction = model.predict(X_dummy)

print("Model Output Prediction:", prediction)