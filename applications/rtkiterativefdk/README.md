# Iterative FDK

Iterative Feldkamp-Davis-Kress reconstruction (Iterative FDK). This application performs an iterative correction of an FDK reconstruction using projection-subset based updates. It supports CPU and CUDA backends and common projector options.

`````{tab-set}

````{tab-item} 3D

## 3D

This script uses the SheppLogan phantom.

```{literalinclude} IterativeFDK3D.sh
```

````
````{tab-item} 2D

## 2D

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.

```{literalinclude} IterativeFDK2D.sh
```

````
