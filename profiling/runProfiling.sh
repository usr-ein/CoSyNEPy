#!/bin/bash
PARAMS="80 200 --psi 3 5 3 --recombine 0.25 --mutate 0.20 --seed 0"

echo "Profiling..."
/usr/bin/env python3 -m cProfile -s time -o profiling/cprofile_report.output src/main.py $PARAMS
/usr/bin/env python3 -m cProfile -s time src/main.py $PARAMS > profiling/cprofile_report.profile
echo "Generating graph..."
/usr/bin/env gprof2dot -n 0.71 -f pstats profiling/cprofile_report.output | dot -Tsvg -o profiling/cprofileGraph.svg
