# recipes-apps/vd100-ps-ma-client/vd100-ps-ma-client.bb
#
# PS XRT application -- MA crossover signal via AIE-ML v2
#
# XRT note: meta-xilinx-tools installs XRT to /opt/xilinx/xrt (NOT /usr).
# The "xrt" and "xrt-dev" packages must be enabled in your image.
# zocl.ko must also be present (enable "zocl" option).
#
# Place in: meta-vd100-v3/recipes-apps/vd100-ps-ma-client/
# Source:   files/  (copy src/main.cpp and CMakeLists.txt here)

SUMMARY = "VD100 PS XRT Client -- MA Crossover AIE Application"
DESCRIPTION = "Sends price ticks to AIE-ML v2 MA crossover graph via XRT, \
               reads back fast_ma / slow_ma / BUY-SELL-HOLD signal."
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COREBASE}/meta/COPYING.MIT;md5=3da9cfbcb788c80a0384361b4de20420"

DEPENDS = "xrt"

SRC_URI = "file://CMakeLists.txt \
           file://vd100-ps-ma-client.cpp"

S = "${WORKDIR}"

inherit cmake

# XRT is installed to /opt/xilinx/xrt in rootfs (not /usr)
# STAGING_DIR_TARGET points to the sysroot for the target architecture
EXTRA_OECMAKE += "-DXRT_ROOT=${STAGING_DIR_TARGET}/opt/xilinx/xrt"

FILES:${PN} += "${bindir}/vd100-ps-ma-client"

# Runtime: XRT libs at /opt/xilinx/xrt/lib + zocl kernel driver
RDEPENDS:${PN} += "xrt zocl"
