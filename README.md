Code - .vs
       .vscode
       .python_scripts
       --extracted_images
       --reduced
       --video_files
       .VOOD
       --data
       --include
       --main
       --src

g++ -std=c++11 -g -O0 .\src\data.cc .\src\data_handler.cc .\src\common.cc .\src\layer.cc .\src\neuron.cc .\src\network.cc -I./include -o .\main\test.exe
--gdb compile

gdb <executable file>
break class function name
print names of variables
