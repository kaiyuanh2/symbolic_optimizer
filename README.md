# symbolic_optimizer
 Experiments for optimizing symbolic operations in ZORRO

## How to compile C++ code
```
g++ -pg -O0 -o executable_name source_code.cpp `pkg-config --cflags --libs ginac`
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