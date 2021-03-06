# Test results

On NCTU CSCC `pp2.cs.nctu.edu.tw`

## Running following commands
```sh
# For one core
$ taskset -c 0 ./bin/cg

# For two core
$ taskset -c 0,1 ./bin/cg

# For four core
$ taskset -c 0,1,2,3 ./bin/cg

```

unit:second

## DATASIZE=LARGE
| Core | Execute | Execute AVG | Speedup | Init  |
|------|---------|-------------|---------|-------|
| 1    | 86.565  |             |         | 5.106 |
|      | 91.585  |             |         | 5.233 |
|      | 89.055  |             |         | 5.193 |
|      | 87.866  | 88.767      | 1.000   | 5.128 |
| 2    | 56.343  |             |         | 4.596 |
|      | 52.165  |             |         | 4.591 |
|      | 56.517  |             |         | 4.593 |
|      | 51.782  | 54.201      | 1.638   | 4.594 |
| 4    | 39.558  |             |         | 4.484 |
|      | 39.276  |             |         | 4.472 |
|      | 40.009  |             |         | 4.474 |
|      | 38.948  | 39.447      | 2.250   | 4.452 |
## DATASIZE=MEDIUMN
| Core | Execute | Execute AVG | Speedup | Init  |
|------|---------|-------------|---------|-------|
| 1    | 1.521   |             |         | 0.504 |
|      | 1.547   |             |         | 0.510 |
|      | 1.521   |             |         | 0.498 |
|      | 1.541   | 1.532       | 1.000   | 0.495 |
| 2    | 1.142   |             |         | 0.506 |
|      | 1.047   |             |         | 0.469 |
|      | 1.100   |             |         | 0.469 |
|      | 1.050   | 1.084       | 1.412   | 0.469 |
| 4    | 0.980   |             |         | 0.492 |
|      | 1.005   |             |         | 0.475 |
|      | 1.107   |             |         | 0.508 |
|      | 1.021   | 1.028       | 1.490   | 0.485 |
## DATASIZE=SMALL
| Core | Execute | Execute AVG | Speedup | Init  |
|------|---------|-------------|---------|-------|
| 1    | 0.293   |             |         | 0.092 |
|      | 0.296   |             |         | 0.093 |
|      | 0.295   |             |         | 0.093 |
|      | 0.290   | 0.293       | 1.000   | 0.092 |
| 2    | 0.159   |             |         | 0.084 |
|      | 0.154   |             |         | 0.084 |
|      | 0.153   |             |         | 0.084 |
|      | 0.150   | 0.154       | 1.902   | 0.086 |
| 4    | 0.102   |             |         | 0.083 |
|      | 0.097   |             |         | 0.084 |
|      | 0.094   |             |         | 0.084 |
|      | 0.087   | 0.095       | 3.084   | 0.087 |
