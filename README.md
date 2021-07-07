Tensor Graph
===

This library implements a simple class to load/evaluate tensorflow models in c++. It is light weight and only depends on lib_tensorflow. It defaults to use GPU, there is no option to switch to CPU yet but it should be trivial to implement.

## Build

1. Download libtensorflow from Google and unzip:

```
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz
```

2. Set environment variable LIBTF_PATH so cmake can find it:

```
export LIBTF_PATH=<path-to-libtensorflow>
```

3. Compile 

```
mkdir build/
cmake . -B build/
make -C build/
```

## Testing

4. Generate the test model (need to have tensorflow installed)

```
python scripts/generate_test_network.py
```

This will create a model file under ```models```. It also prints the desired output from the model. Please note this is different each time you generate the model, due to the random initialization of the weights.

5. Run the unit test

```
./build/example/reading_keras_model ./models/test_model.pb
```

You should now be able to verify that the output from the c++ program is the same as that from the python script. 

## Use it in your program

Apart from the source files, you should also include the camke file ```FindTensorFlow.cmake``` into your project. You also need to repeat step 2 when building your own software.

Please refer to ```scripts/generate_test_network.py``` as how to generate concrete functions from your own model. You should keep an eye on the names of the input and output node(s).

```example/reading_keras_model.cpp``` demonstrates the basic usage of this library. It should be pretty straight forward to adapt it to your needs.

## Loading custom operations

Some models might include custom operations (e.g. PointNet++) as separate DLLs. They could be loaded with ```load_custom_operators```. This function is not yet fully tested but should work in most cases. 