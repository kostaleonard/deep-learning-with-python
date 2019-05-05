# py_deep_learning

Exercises and models from the book "Deep Learning with Python" by Francois Chollet.

## AMD GPU Support (MacOS)

Tensorflow and Keras should work easily already with NVIDIA GPUs, but AMD GPUs
need some extra configuration.
You need to run python from within the virtual environment in order to take
advantage of AMD GPUs on your machine. However, this is only the first step.
Running from outside the virtual environment, my
Mac finishes ch2/mnist.py in 68.720 seconds. From inside the environment, it
finishes in 18.580 seconds.

*To actually activate the GPU*, you need to insert the following lines of
code at the start of your program:

```
import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
```

If successful, you should see something like the following as output:

```
Using plaidml.keras.backend backend.
INFO:plaidml:Opening device "metal_amd_radeon_pro_vega_20.0"
```

*For some networks, the GPU will be slower than the CPU!* On ch2/mnist.py, for
example, turning on my GPU leads to a time of 33.630 seconds. But on gpu_config.py,
I get a speedup of about 8 times when I run with a GPU: 2.177 seconds vs. 16.483
seconds when `os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'` is commented
out. Speedup tends to be best when the network relies on convolutional layers.
Maybe recurrent layers, too, but I haven't tested that just yet. Try out both on
your networks and go with whatever is fastest.
