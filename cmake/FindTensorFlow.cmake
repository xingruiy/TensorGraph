###############################################################################
# Find TensorFlow
#
#     find_package(TensorFlow)
#
# Variables defined by this module:
#
#  TENSORFLOW_FOUND                True if OpenNI was found
#  TF_INCLUDE_DIRS         The location(s) of OpenNI headers
#  TF_LIBRARIES                 Libraries needed to use OpenNI

find_path(TF_INCLUDE_DIR 
NAME tensorflow
HINTS "$ENV{LIBTF_PATH}"
PATH_SUFFIXES include)

find_library(TF_LIBRARY
NAME libtensorflow.so
HINTS "$ENV{LIBTF_PATH}"
PATH_SUFFIXES lib Lib Lib64)

if(TF_INCLUDE_DIR AND TF_LIBRARY)
  set(TF_INCLUDE_DIRS ${TF_INCLUDE_DIR})
  unset(TF_INCLUDE_DIR)
  mark_as_advanced(TF_INCLUDE_DIRS)

  set(TF_LIBRARIES ${TF_LIBRARY})
  unset(TF_LIBRARY)
  mark_as_advanced(TF_LIBRARIES)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow
  FOUND_VAR TENSORFLOW_FOUND
  REQUIRED_VARS TF_LIBRARIES TF_INCLUDE_DIRS
)

if(TENSORFLOW_FOUND)
  message(STATUS "TensorFlow found (include: ${TF_INCLUDE_DIRS}, lib: ${TF_LIBRARIES})")
endif()