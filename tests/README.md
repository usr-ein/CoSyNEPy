# CoSyNE Python
## An (hopefully clean) implementation of [Accelerated Neural Evolution through Cooperatively Coevolved Synapses](https://pdfs.semanticscholar.org/966e/41903b4aff42601a188bd7b26d71ef120d11.pdf) in Python3

### Current state of developpment
Right now I'm still implementing the base algorithm. The next phase will be profiling it to imporve its speed.

### Role of each directory
* cache: Preprocessed datasets that don’t need to be re-generated every time you perform an analysis.
* config: Configuration settings for the project
* data: Raw data files.
* preprocessing: Preprocessing data munging *scripts*, the outputs of which are put in cache.
* src: Statistical analysis and ML trainer scripts.
* diagnostics: Scripts to diagnose data sets for corruption or outliers.
* doc: Documentation written about the analysis.
* graphs: Graphs created from analysis.
* lib: Helper library functions but not the core statistical analysis.
* logs: Output of scripts and any automatic logging.
* profiling: Scripts to benchmark the timing of your code.
* reports: Output reports and content that might go into reports such as tables.
* tests: Unit tests and regression suite for your code.
* testing: Notebooks used for testing individual algorithm before definitive implementation.
* README.md: Notes that orient any newcomers to the project.
* TODO.md: list of future improvements and bug fixes you plan to make.
