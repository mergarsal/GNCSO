# GNCSO Library: GNC robust framework 

Header-based library for robust estimation: 
Graduated Non-Convexity + Black-Rangarajan duality 

**Authors:** [Mercedes Garcia-Salguero](http://mapir.uma.es/mapirwebsite/index.php/people/290), [Javier Gonzalez-Jimenez](http://mapir.isa.uma.es/mapirwebsite/index.php/people/95-javier-gonzalez-jimenez)

**License:** TODO

## Dependences 
* Eigen 
 ```
        sudo apt install libeigen3-dev
 ```

* Optimization (own fork)
 ```
        https://github.com/mergarsal/Optimization.git

 ```


## Build
```
git clone --recursive https://github.com/mergarsal/GNCSO.git
cd GNCSO

mkdir build & cd build 

cmake .. 

make -jX

```

The compiled examples should be inside the `examples` directory. Run: 
```
        ./examples/GNC_rotation_example
```
 


## Install 
In `build` folder: 
```
        sudo make install
```

We also provide the uninstall script: 
```
        sudo make uninstall
```


## How to use the library in your project
        
In your project, add the package: 
```
        find_package(GNCSO REQUIRED)
```


In your executable, add the library in 
```
        target_link_libraries(XXX gncso)
```


See folder `example_install` for an example. 



