cmake_minimum_required(VERSION 3.16)

project(linux-ai-vision
        VERSION 0.1.0
        DESCRIPTION "Real-time face detection and AI-estimated expression classification on Linux"
        LANGUAGES CXX)

enable_testing()

add_subdirectory(cpp)
