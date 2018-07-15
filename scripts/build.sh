# Moves to the correct Dockerfile Directory to build Image
# This exists since I am too lazy to cd to directory everytime, and I like 
# performing all commands from the root of the project

cd $1
if [[ $? == 0  ]]; then
    ./build.sh
fi