# LLM (C++)

LLM C++ project implements various LLM models in C++ using the AIX machine learning framework.

### Overview

GPT2 - Inference only implementation utilizing OpenAI weights with parameters of 124M, 355M, 774M, and 1.5B.

### Quick Start

Assuming you have already built the project explained in the following section, and you are currently in the project’s root folder:

```bash
# Download OpenAI GPT2 model weights only once. Options: 124M, 355M, 774M and 1558M

$ cd Resources
$ python downloadGPT2.py 124M
$ cd ..

# Give it a try, assuming the binary folder name is product-rel.

$ cd product-rel
$ ./GPT2 --prompt="What do you know about artificial intelligence?" --model=124M --device=CPU
```

NOTE: If you have an Apple Silicon hardware, use --device=MCS for GPU acceleration.

# Project Build Instructions

Follow the following steps to build the project and make it deployment ready.

Currently, it has been built and tested on macOS Sonoma with no issues.

---

### Step 1: Build Externals

This step will build external libraries.

```bash
cd Externals
./build_aix.sh
./build_docopt.sh
```

### Step 2: Build Targets

This step will build all binaries and deploy into a specific folder. Assuming you are in the root folder of the project.

```bash
./build.sh release product-rel
```

After the successful build, all target binaries will be deployed into the product-rel folder.

Note: Run the build.sh file without parameters to see all options.

# Citation

If you find the project useful in your research, please consider citing it and use the following BibTex entry:

```
@software{LLMCPP2024,
   author = {Arkin Terli},
   title = {{LLM-C++}: Experimental LLM (Large Language Model) implementations in C++.},
   url = {https://github.com/godrays/llm},
   version = {0.0.0},
   year = {2024},
}
```

# License

Copyright © 2024 - Present, Arkin Terli. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of Arkin Terli nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
