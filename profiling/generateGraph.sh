#!/bin/bash
echo "Generating graph..."
/usr/bin/env gprof2dot -n 5.0 -f pstats profiling/cprofile_report.output | dot -Tsvg -o profiling/cprofileGraph.svg
