#!/bin/bash
echo "Profiling..."
/usr/bin/env python3 -m cProfile -s time -o profiling/cprofile_report.output src/main.py
/usr/bin/env python3 -m cProfile -s time src/main.py > profiling/cprofile_report.profile
echo "Generating graph..."
/usr/bin/env gprof2dot -n 0.71 -f pstats profiling/cprofile_report.output | dot -Tsvg -o profiling/cprofileGraph.svg
