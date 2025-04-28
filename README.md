Organization:  
```
project/
├── build/
├── data/
│   ├── graph.mtx
│   └── communities.txt
├── include/
│   ├── common.hpp
|   ├── printer.hpp
│   ├── R_spanner_kernels.cuh
│   └── R_spanner_helper.hpp   
├── src/
│   ├── R_spanner_helper.cpp
│   ├── R_spanner_kernels.cu
│   ├── printer.cpp
│   └── R_spanner.cu (main)
├── tests/
├── tools/
│   ├── community_detection
│   └── plot
├── CMakeLists.txt
```



Build:  

```
mkdir build  
cd build      
cmake ..  
make -j     # or (for powershell), cmake --build . --config Release
```

Build with tests ON (for powershell):  
``` 
cd build  
cmake -DENABLE_TESTS=ON ..
cmake --build . --config Release
ctest --verbose  -C Release
```

Debug:
```
cd build 
cmake -DENABLE_TESTS=OFF ..
cmake --build . --config Debug
.\Debug\r_spanner_cuda.exe -g ..\data\test_graph.mtx -c ..\data\community_test_graph.txt -t 4 -t 1 -t 3 -t 5 -t 2
```

Run:  
```
./r_spanner_cuda -g ../data/test_graph.mtx -c ../data/community_test_graph.txt  -t 4 -t 1 -t 3 -t 5 -t 2 
or (for powershell), .\Release\r_spanner_cuda.exe -g ..\data\test_graph.mtx -c ..\data\community_test_graph.txt  -t 4 -t 1 -t 3 -t 5 -t 2
```
-g graph.mtx -> input graph file in .mtx format  
-c communities.txt -> node-to-community mapping  
-t -> target community IDs (repeatable)  



Rebuild:  
```
cd build      
rm -rf *    # or (for powershell), Remove-Item * -Recurse -Force  
cmake ..     
make -j     # or (for powershell), cmake --build . --config Release  
```

