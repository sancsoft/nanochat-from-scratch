import os
import sys
import subprocess
import ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
cpp_path = os.path.join(HERE, "tokenizer.cpp")

# Linux only
if not sys.platform.startswith("linux"):
    raise RuntimeError("This script supports Linux only")

lib_name = "tokenizer.so"
lib_path = os.path.join(HERE, lib_name)

# Build with ICU (requires: sudo apt-get install -y libicu-dev)
compile_cmd = [
    "c++",
    "-std=c++17",
    "-shared",
    "-fPIC",
    cpp_path,
    "-o",
    lib_name,
    "-licui18n",
    "-licuuc",
    "-licudata",
]

print("Compiling:", " ".join(compile_cmd))
subprocess.check_call(compile_cmd, cwd=HERE)

lib = ctypes.CDLL(lib_path)

# int train_tokenizer(const char* input_utf8, const char* output_path, int32_t vocab_count);
lib.train_tokenizer.argtypes = (
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int32,
)
lib.train_tokenizer.restype = ctypes.c_int

test_string = "../data"
output_path = os.path.join(HERE, "output.txt")
VOCAB_COUNT = 256

result = lib.train_tokenizer(
    test_string.encode("utf-8"),
    output_path.encode("utf-8"),
    VOCAB_COUNT,
)

print("Result code:", result)
