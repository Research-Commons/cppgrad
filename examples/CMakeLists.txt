file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS *.cpp)

foreach(file ${EXAMPLE_SOURCES})
    get_filename_component(example ${file} NAME_WE)
    add_executable(${example} ${file})
    target_link_libraries(${example} PRIVATE cppgrad)
endforeach()