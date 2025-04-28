## Prepare communities  


### Convert .mtx to .metis:  
----------------------------  
```
g++ -O3 -o op_mtx2metis mtxToMetis_skipLines.cpp  
./op_mtx2metis graphFile.mtx > graphFile.metis  
```

### Community detection:  
-----------------------------  
we use parallel implementation of the well-known Louvain method from https://networkit.github.io/dev-docs/notebooks/User-Guide.html#Community-Detection  
```
python find_community_usingNetworkit.py -f graphFile.metis
```



### Use below command  to make it 1-indexed:    
----------------------------------------  
NOTE: The community ID starts from 0  
Convert it to 1-indexed for RS detection code  
```
awk '{print $1+1}' communties_graphFile.txt > communties_graphFile_updated.txt
```


### Get top 10 communities with highest number of vertices:  
-------------------------------------------------------  
```
cat communties_graphFile_updated.txt   | sort -n | uniq -c | sort -nr | head -10
```


