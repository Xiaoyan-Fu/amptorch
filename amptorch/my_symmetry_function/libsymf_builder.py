import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef(
    """int calculate_sf(double **, double **, double **,
                                    int *, int, int*, int,
                                    int**, double **, int,
                                    double**);"""
)
ffibuilder.set_source(
    "amptorch.my_symmetry_function._libsymf",
    '#include "calculate_sf.h"',
    sources=[
        "amptorch/my_symmetry_function/calculate_sf.cpp",
        "amptorch/my_symmetry_function/symmetry_functions.cpp"
    ],
    source_extension=".cpp",
    include_dirs=["amptorch/my_symmetry_function/"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
