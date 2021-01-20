# GNCSO Library: GNC robust framework 

Header-based library for robust estimation: 
Graduated Non-Convexity + Black-Rangarajan duality. 
Code associated to the paper: 
[Fast and Robust Relative Pose Estimation for Calibrated Cameras](ADD ARXIV) 

**Authors:** [Mercedes Garcia-Salguero](http://mapir.uma.es/mapirwebsite/index.php/people/290), [Javier Gonzalez-Jimenez](http://mapir.isa.uma.es/mapirwebsite/index.php/people/95-javier-gonzalez-jimenez)

**License:** [GPLv3](https://raw.githubusercontent.com/mergarsal/GNCSO/main/LICENSE.txt)


If you use this code for your research, please cite:
```
ADD OURS
```

Note: This library wraps around the [Optimization library](https://github.com/david-m-rosen/Optimization.git)
```
        @article{rosen2019se,
          title={SE-Sync: A certifiably correct algorithm for synchronization over the special Euclidean group},
          author={Rosen, David M and Carlone, Luca and Bandeira, Afonso S and Leonard, John J},
          journal={The International Journal of Robotics Research},
          volume={38},
          number={2-3},
          pages={95--125},
          year={2019},
          publisher={Sage Publications Sage UK: London, England}
        }
```

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
        find_package(gncso REQUIRED)
```


In your executable, add the library in 
```
        target_link_libraries(XXX gncso)
```


See folder `example_install` for an example. 



