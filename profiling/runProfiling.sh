#!/bin/bash
PARAMS="80 -m 200 --psi 3 5 3 --recombine 0.25 --mutate 0.20 --seed 0"

echo "Profiling with parameters $PARAMS"
/usr/bin/env python3 -m cProfile -s time -o profiling/cprofile_report.output src/main.py $PARAMS
/usr/bin/env python3 -m cProfile -s time src/main.py $PARAMS > profiling/cprofile_report.profile
