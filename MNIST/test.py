import numpy as np

n_layers = 5
layer_size = 1000

encoder_layers = np.zeros(n_layers) # Initialization
for n in range(0,n_layers):
    layer_size = int(layer_size/2) # The new layer has a half-size of the previous layer
    encoder_layers[n] = layer_size # Save the layer size
print(encoder_layers)