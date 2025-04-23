# symbolic_optimizer
Experiments for optimizing symbolic operations in ZORRO, includes AI-generated code

## How to compile C++ code
```
g++ -pg -O0 -o executable_name source_code.cpp `pkg-config --cflags --libs ginac`
```

## How to compile Python Module (ginac_module.cpp with C++ 11 and pybind11)
```
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) ginac_module.cpp -o ginac_module$(python3-config --extension-suffix) -lginac -lcln
```

## How to run profiler (Callgrind)
-pg flag was intended for GNU gprof but has problems in counting function exections and running time.
Use valgrind instead:
```
valgrind --tool=callgrind ./executable_name
```

Remember the pid of valgrind, after valgrind finishes:
```
callgrind_annotate callgrind.out._pid_
```

! callgrind.out.insurance, callgrind.out.mines, and callgrind.out.heart are provided as my original running results !

## How to run profiler (gprof for Jupyter Notebook)
Follow instructions inside notebooks.