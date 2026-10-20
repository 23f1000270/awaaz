idf_component_register(SRCS "main.c" "../../csrc/awaaz.c"
                       INCLUDE_DIRS "." "../../csrc")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
