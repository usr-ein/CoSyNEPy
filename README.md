# CoSyNE Python
## An (hopefully clean) implementation of [Accelerated Neural Evolution through Cooperatively Coevolved Synapses](https://pdfs.semanticscholar.org/966e/41903b4aff42601a188bd7b26d71ef120d11.pdf) in Python3

### Current state of developpment
~~Right now I'm still implementing the base algorithm. The next phase will be profiling it to imporve its speed.~~
I'm done implementing the base structure, I've tested it on the rosenbrock function and it works ! I've done some profiling and the code now differs a bit from the paper's description but it's faster and does the same. I've made it a bit user friendly as well with a nice CLI.
The next phase will be to evaluate it on others problems using OpenAI's Gym. Also I should get it to work on multicore because right now something (not numpy and not the NeuralNetwork class) is limiting it to one core.
Also I should do something to make the evaluation method editable from the outside.

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
