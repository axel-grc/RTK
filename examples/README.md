C++ / Python examples
========

This section provides a collection of C++ and Python code examples 
to demonstrate how to effectively use RTK in various applications.

A key example is [FirstReconstruction](./FirstReconstruction/README.md), 
a generic starting point for users new to RTK. This example demonstrates 
the fundamental steps of performing a reconstruction, making it an ideal 
entry point for those beginning with RTK.
It showcases how to set up an RTK pipeline, configure parameters, 
and execute a basic reconstruction process.

Additionally, there are examples which use CUDA for GPU-accelerated computations, 
providing efficient implementations for high-performance applications.

While C++ in RTK typically uses an object-oriented approach with setters and manual pipeline connections, the python examples provide a more concise functional style. Functions like rtk.constant_image_source let users pass parameters directly as arguments.
However, some advanced workflows, such as real-time or on-the-fly reconstruction, require more control over the pipeline. In these cases, object-oriented methods like .SetInput(), .Update(), and .DisconnectPipeline() are necessary to manage the pipeline dynamically. So, both styles are complementary: use functional style for simplicity and object-oriented for fine-grained control.

```{toctree}
:maxdepth: 1

./FirstReconstruction/README.md
./WaterPreCorrection/README.md
./InlineReconstruction/README.md
./AddNoise/README.md
```
