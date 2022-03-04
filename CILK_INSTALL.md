System: OSX 12.2.1 (21D62)
Processor: M1 Max
Apple Clang 13

Here are the steps to reproduce building from source the Cilk Compiler.


```
git clone git@github.com:OpenCilk/opencilk-project.git
cd opencilk-project
git clone git@github.com:OpenCilk/cheetah.git
git clone https://github.com/OpenCilk/productivity-tools
mv productivity-tools cilktools
cd llvm
mkdir build
cd build
cmake -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;compiler-rt" -DCMAKE_BUILD=RelWithDebInfo -DLLVM_ENABLE_RUNTIMES="cheetah;cilktools" ../
make -j 10
```

Inside the `build` directory I now set my environment variable to point to the compiler frontend:

```
export PATH=$(pwd)/bin:$PATH
```

https://blog.nuullll.com/2021/05/15/building-llvm-project-with-ninja-on-macos.html

Now, you must either perform `SDKROOT=`xcrun --show-sdk-path` make` everytime you compile the program or alternatively add this flag to the makefile:

Or add this to your `~/.zshrc` file:

```
export PATH=/Users/alejandro/Downloads/opencilk-project/llvm/build/bin:$PATH
export SDKROOT=$(xcrun --show-sdk-path --sdk macosx)
```