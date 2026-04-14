# vd100-ps-ma-client

**PS userspace XRT application — Moving Average crossover via AIE-ML on XCVE2302**

Cortex-A72 host application that drives the [`vd100-aie-pipeline`](https://github.com/adilsondias-engineer/vd100-aie-pipeline) hardware over XRT. Feeds price data to the AIE-ML MA crossover kernel via HLS DMA kernels and reads BUY/SELL/HOLD signals from the output buffer.

Part of the [versal-ai-edge-vd100-linux](https://github.com/adilsondias-engineer/versal-ai-edge-vd100-linux) ecosystem — deployed as a Yocto recipe in [`meta-vd100_v3`](https://github.com/adilsondias-engineer/meta-vd100_v3).

---

## Pipeline

```
PS DDR (int32[] prices)
    │
    │ XRT BO → sync to device
    ▼
mm2s (HLS) ──AXI4-Stream──► AIE graph (mygraph)
                             MA crossover kernel
                             FAST=10 / SLOW=50
                             56 samples per iteration
                                       │
                           AXI4-Stream ▼
                           s2mm (HLS) ──► PS DDR (int32[3])
                                              │
                                              ▼
                                 fast_ma / slow_ma / signal
                                 1=BUY  -1=SELL  0=HOLD
```

### Hardware path

```
PS DDR4 → NoC (M_AXI_FPD) → mm2s PL kernel → AXI4-Stream → AIE array
AIE array → AXI4-Stream → s2mm PL kernel → NoC → PS DDR4
```

---

## Hardware

- **Board**: ALINX VD100 (XCVE2302-SFVA784-1LP-E-S)
- **xclbin**: `aie.xclbin` from [`vd100_ma_system_project`](https://github.com/adilsondias-engineer/vd100_ma_system_project)
- **xclbin UUID**: `0f5096a5-b416-a54c-8035-9efc0e394fdc`
- **AIE kernel**: [`vd100-aie-ma-crossover`](https://github.com/adilsondias-engineer/vd100-aie-ma-crossover) — `ma_crossover<50>`
- **Timing closure**: WNS 4.217ns, WHS 0.018ns
- **OS**: Yocto Scarthgap / linux-xlnx 6.12.40 / XRT 2025.2

---

## AIE Kernel Constants

| Constant | Value | Notes |
|----------|-------|-------|
| `FAST_MA_PERIOD` | 10 | Fast moving average window |
| `SLOW_MA_PERIOD` | 50 | Slow moving average window (= ADF margin M) |
| `BLOCK_SIZE` | 56 | int32 samples per AIE iteration |
| `NUM_MARGIN_SAMPLES` | 56 | Must equal BLOCK_SIZE — 224B is a multiple of 32 |
| Output per block | 3 × int32 | `[fast_ma, slow_ma, signal]` |
| Signal encoding | int32 | `1=BUY`, `-1=SELL`, `0=HOLD` |

---

## HLS Kernel Signatures

```cpp
// Confirmed via xclbinutil --info
void mm2s(ap_int<32>* mem, hls::stream<ap_axis<32,0,0,0>>& s, int size)
void s2mm(ap_int<32>* mem, hls::stream<ap_axis<32,0,0,0>>& s, int size)
```

Both kernels: `mem = arg[0]` → `group_id(0)` for XRT BO allocation.  
`size` is in **bytes** — pass `count * sizeof(int32_t)`.

---

## XRT API — Critical Notes

### Kernel and BO construction (XRT 2025.2)

```cpp
// Use hw_context API — device+uuid pattern is deprecated in 2025.2
xrt::device device(0);
auto uuid = device.register_xclbin(xrt::xclbin{xclbin_path});
xrt::hw_context ctx(device, uuid, xrt::hw_context::access_mode::exclusive);

auto mm2s_k = xrt::kernel(ctx, "mm2s:{mm2s_1}");
auto s2mm_k = xrt::kernel(ctx, "s2mm:{s2mm_1}");

// BOs allocated via hw_context
auto in_bo  = xrt::bo(ctx, input_bytes,  xrt::bo::flags::none, mm2s_k.group_id(0));
auto out_bo = xrt::bo(ctx, output_bytes, xrt::bo::flags::none, s2mm_k.group_id(0));
```

### Graph construction via hw_context (XRT 2025.2)

```cpp
// xrt::graph must be constructed from hw_context, not (device, uuid, name)
xrt::graph graph(ctx, "mygraph");
```

### Execution order — s2mm before mm2s

```cpp
graph.reset();

// s2mm FIRST — sink must be ready before source fires
xrt::run s2mm_run = s2mm_k(out_bo, nullptr,
    static_cast<unsigned int>(output_bytes));

// mm2s SECOND — fires data into AIE
xrt::run mm2s_run = mm2s_k(in_bo, nullptr,
    static_cast<unsigned int>(input_bytes));

graph.run(num_blocks);  // N iterations = N × 56 samples
graph.wait();
s2mm_run.wait();
mm2s_run.wait();
```

If mm2s fires before s2mm is ready, the AIE output stream backs up and the graph stalls.

### Output indexing

```cpp
out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
const int32_t* out = out_bo.map<const int32_t*>();

// Per block b (0-indexed):
int32_t fast_ma = out[b * 3 + 0];
int32_t slow_ma = out[b * 3 + 1];
int32_t signal  = out[b * 3 + 2];  // 1=BUY, -1=SELL, 0=HOLD
```

---

## Usage

```bash
# Golden test vector (built-in — validates against golden.txt)
./vd100-ps-ma-client /lib/firmware/xilinx/aie.xclbin

# Custom price file (one int32 per line, must be multiple of 56 samples)
./vd100-ps-ma-client /lib/firmware/xilinx/aie.xclbin prices.txt
```

---

## Expected Output — Golden Test Vector

```
+---------+----------+----------+--------+
|  Block  | fast_ma  | slow_ma  | Signal |
+---------+----------+----------+--------+
|      1  |    5000  |    5000  |  HOLD  |
|      2  |    4990  |    4990  |  HOLD  |
|      3  |    5051  |    5002  |  BUY   |  ← crossover triggered
|      4  |    5600  |    5600  |  HOLD  |
+---------+----------+----------+--------+

[DONE] 4 block(s) processed via AIE-ML v2 (XCVE2302)
```

Block 3 BUY: fast MA (last 10 samples) crosses above slow MA (last 50) — correct crossover detection confirmed on hardware.

---

## XRT Performance — Lifecycle Timing (XCVE2302)

Measured on VD100 with XCVE2302, XRT 2025.2:

```
device + register_xclbin:   ~2ms
hw_context creation:         ~18-21ms   ← load_hw_axlf (programs FPGA hardware)
kernel construction (×2):   ~14-15ms   ← DRM_IOCTL_ZOCL_OPEN_CU_CTX
BO allocation (×2):         ~0.4-1.9ms
BO sync (input):            ~19µs
graph construction:          ~2.5ms    ← DRM_IOCTL_ZOCL_OPEN_GRAPH_CTX
actual AIE run (4 blocks):   ~1ms
─────────────────────────────────────
Total (first call):          ~47ms
```

### Why hw_context creation is expensive every call

Investigation of XRT source (`shim.cpp`, commit `a4d60af`) reveals:

```cpp
// AMD comment in prepare_hw_axlf():
// "This flag is added to support force xclbin download eventhough same
//  xclbin is already programmed. This is required for aie reset/reinit
//  in next run. Aie is not clean after first run. We need to work with
//  aie team to figureout a solution to reset/reinit AIE in second run."

auto force_program = xrt_core::config::get_force_program_xclbin() ||
                     buffer->m_header.m_actionMask & AM_LOAD_PDI;
```

Versal xclbins always set `AM_LOAD_PDI` — `force_program` is always true. AMD intentionally forces full hardware reprogram on every `hw_context` creation because AIE tile state is not cleanly reset otherwise. This is an acknowledged open problem.

**Consequence:** This client is optimal for batch/bulk workloads. For per-tick streaming use cases (one block per call), SW MA on A72 (~1440ns) outperforms AIE by 30×. See [`vd100-ma-trading-signal`](https://github.com/adilsondias-engineer/vd100-ma-trading-signal) for the production streaming implementation.

### Optimal use case

This client processes multiple blocks in a single XRT session — amortising the ~35ms init cost over N blocks:

```
N=1    block:  ~47ms   (init dominates)
N=10   blocks: ~57ms   (~5.7ms per block)
N=100  blocks: ~150ms  (~1.5ms per block)
N=1000 blocks: ~1.07s  (~1.07ms per block ≈ pure AIE compute)
```

For large N the XRT overhead becomes negligible and pure AIE throughput dominates.

---

## Address Map (Vivado Address Editor)

| IP | Base Address | Range |
|----|-------------|-------|
| MyLEDIP | `0xA400_0000` | 4K |
| mm2s_1 / s_axi_control | `0xA401_0000` | 64K |
| s2mm_1 / s_axi_control | `0xA402_0000` | 64K |
| DDR (mm2s via NoC) | `0x0000_0000` | 2G |
| DDR (s2mm via NoC) | `0x0000_0000` | 2G |

---

## Build

### Yocto (recommended — deployed as rootfs recipe)

```bash
# On yoctoBuilder
source /work/yocto/setupsdk
bitbake vd100-ps-ma-client

# Or include in full image
bitbake edf-linux-disk-image
```

Recipe lives in `meta-vd100_v3/recipes-apps/vd100-ps-ma-client/`.

### Manual cross-compile (Yocto SDK)

```bash
source /opt/vd100/sdk/environment-setup-cortexa72-cortexa53-vd100-linux
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DXRT_ROOT=${SDKTARGETSYSROOT}/opt/xilinx/xrt
make -j$(nproc)
scp vd100-ps-ma-client root@vd100:~/
```

### Deploy xclbin

```bash
# Copy aie.xclbin from build artefacts to VD100
scp aie.xclbin root@vd100:/lib/firmware/xilinx/
```

The xclbin is produced by the `vd100_ma_system_project` Vitis system build. It contains:
- `mm2s` HLS kernel
- `s2mm` HLS kernel
- AIE graph (`mygraph`) with `ma_crossover<50>` kernel
- CDO partitions: `aie_dev_part` + `aie_image` ← **required for AIE tiles to ungate**

---

## Key Lessons Learned

### BOOT.BIN must include AIE CDO partitions

Without `aie_dev_part` and `aie_image` partitions in BOOT.BIN, all AIE tiles remain permanently `clock_gated` and the pipeline produces no output. Fixed via `meta-vd100_v3` bbappend to `xilinx-bootbin` recipe:

```bitbake
# xilinx-bootbin_1.0.bbappend
BOOTBIN_DEPENDS:append = " aie.cdo.device.partition.reset.bin aie.merged.cdo.bin"
```

### XRT 2025.2 include path quirks (Yocto sysroot)

```cmake
# xrt_graph.h is NOT in the standard xrt/ path in Yocto sysroot
# Use: /opt/xilinx/xrt/include/xrt/xrt_graph.h
# Or: #include <xrt/xrt_aie.h>  (experimental, includes graph)
```

### xrt::graph constructor — use hw_context overload

```cpp
// CORRECT (XRT 2025.2)
xrt::graph graph(ctx, "mygraph");

// WRONG — causes segfault on XCVE2302
xrt::graph graph(device, uuid, "mygraph");
```

### JTAG must be disconnected during XRT runtime

Leaving JTAG connected during `graph.run()` causes AIE reinitialisation — tiles reset to zero output. Disconnect JTAG before running.

---

## Repository Structure

```
vd100-ps-ma-client/
├── src/
│   └── main.cpp           # XRT host application
├── include/
│   └── golden.h           # Built-in golden test vector (4 × 56 samples)
├── CMakeLists.txt
└── README.md
```

---

## Related Repositories

| Repo | Description |
|------|-------------|
| [`vd100-aie-pipeline`](https://github.com/adilsondias-engineer/vd100-aie-pipeline) | Vivado block design + xclbin build |
| [`vd100-aie-ma-crossover`](https://github.com/adilsondias-engineer/vd100-aie-ma-crossover) | AIE-ML v2 MA kernel source |
| [`vd100_ma_system_project`](https://github.com/adilsondias-engineer/vd100_ma_system_project) | Vitis system project — produces aie.xclbin |
| [`vd100_platform`](https://github.com/adilsondias-engineer/vd100_platform) | Vitis platform (reusable) |
| [`vd100-ma-trading-signal`](https://github.com/adilsondias-engineer/vd100-ma-trading-signal) | Production streaming app (Binance → Hardhat) |
| [`meta-vd100_v3`](https://github.com/adilsondias-engineer/meta-vd100_v3) | Yocto layer — deploys this app to VD100 rootfs |
| [`versal-ai-edge-vd100-linux`](https://github.com/adilsondias-engineer/versal-ai-edge-vd100-linux) | VD100 Linux bring-up root repo |

---

*Adilson de Souza Dias | April 2026 | adilsondias-engineer*
