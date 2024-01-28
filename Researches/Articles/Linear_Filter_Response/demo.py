import numpy as np
import matplotlib.pyplot as plt

# Create a low-pass FIR filter 
fir_coeffs = [0.1, 0.2, 0.4, 0.2, 0.1] 

# Input signal - sine wave at 10 Hz
input_signal = np.sin(2 * np.pi * 10 * np.arange(1000) / 1000)

# Apply filter to input signal using convolution 
output_signal = np.convolve(input_signal, fir_coeffs)

# Plot input and output signals
plt.subplot(2, 1, 1)
plt.plot(input_signal)
plt.title('Input Signal')

plt.subplot(2, 1, 2) 
plt.plot(output_signal)
plt.title('Output Signal')

plt.tight_layout()
plt.show()