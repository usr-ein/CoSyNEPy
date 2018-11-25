#!/bin/bash

echo "Profiling on the main profiling script"
/usr/bin/env python3 -m cProfile -s time -o profiling/cprofile_report.output src/mainProfiling.py
/usr/bin/env python3 -m cProfile -s time src/mainProfiling.py > profiling/cprofile_report.profile