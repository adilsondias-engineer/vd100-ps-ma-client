cmake_minimum_required(VERSION 3.18)
project(vd100-ps-ma-client VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- XRT install path --------------------------------------------------------
# Yocto XRT recipe (meta-xilinx-tools) installs to /opt/xilinx/xrt — NOT /usr.
# This is set in the XRT recipe config:
#   "Enable xrt and xrt-dev: installs to /opt/xilinx/xrt in rootfs"
#
# Yocto recipe passes:
#   EXTRA_OECMAKE += "-DXRT_ROOT=${STAGING_DIR_TARGET}/opt/xilinx/xrt"
#
# Native fallback (running directly on VD100):
#   /opt/xilinx/xrt/include  /opt/xilinx/xrt/lib
# -----------------------------------------------------------------------------

if(DEFINED XRT_ROOT)
    set(XRT_INCLUDE_DIR "${XRT_ROOT}/include")
    set(XRT_LIB_DIR     "${XRT_ROOT}/lib")
else()
    # Native build on VD100 — XRT installed at /opt/xilinx/xrt
    set(XRT_ROOT        "/opt/xilinx/xrt")
    set(XRT_INCLUDE_DIR "${XRT_ROOT}/include")
    set(XRT_LIB_DIR     "${XRT_ROOT}/lib")
endif()

message(STATUS "XRT_ROOT:        ${XRT_ROOT}")
message(STATUS "XRT_INCLUDE_DIR: ${XRT_INCLUDE_DIR}")
message(STATUS "XRT_LIB_DIR:     ${XRT_LIB_DIR}")

# --- Target ------------------------------------------------------------------
add_executable(vd100-ps-ma-client vd100-ps-ma-client.cpp)

target_include_directories(vd100-ps-ma-client PRIVATE
    "${XRT_INCLUDE_DIR}"
    "${XRT_INCLUDE_DIR}/xrt"
)

target_link_directories(vd100-ps-ma-client PRIVATE
    "${XRT_LIB_DIR}"
)

target_link_libraries(vd100-ps-ma-client PRIVATE
    xrt_coreutil    # xrt::device, xrt::bo, xrt::kernel, xrt::run
    xrt_core        # xrt::graph (AIE runtime), xrt::hw_context
    uuid
    pthread
    dl
)

target_compile_options(vd100-ps-ma-client PRIVATE
    -Wall -Wextra -O2
)

# --- Install -----------------------------------------------------------------
install(TARGETS vd100-ps-ma-client
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)