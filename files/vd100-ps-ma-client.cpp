/**
 * vd100-ps-ma-client — PS XRT Application
 * VD100 (XCVE2302) | MA Crossover via AIE-ML v2
 * Built against XRT BRANCH="2025.2"
 *
 * Pipeline:
 *   PS DDR → mm2s (HLS) → AIE graph (mygraph) → s2mm (HLS) → PS DDR
 *
 * Kernel signatures (both mem = arg[0] → group_id(0)):
 *   void mm2s(ap_int<32>* mem, hls::stream<ap_axis<32,0,0,0>>& s, int size)
 *   void s2mm(ap_int<32>* mem, hls::stream<ap_axis<32,0,0,0>>& s, int size)
 *
 * AIE graph (mygraph):
 *   Input:  int32 price ticks — BLOCK_SIZE=56 samples per iteration
 *   Output: int32[3] per iteration — { fast_ma, slow_ma, signal }
 *           signal: 1=BUY  -1=SELL  0=HOLD
 *
 * XRT 2025.2 API notes:
 *   - Use xrt::xclbin + device.register_xclbin() + xrt::hw_context
 *   - xrt::kernel(hw_context, name)  — NOT (device, uuid, name) [deprecated]
 *   - xrt::bo(device, size, group_id) — device-based still valid in 2025.2
 *   - xrt::graph(device, uuid, name)  — AIE graph API unchanged
 *   - graph.run(N) runs N iterations; graph.wait() blocks until done
 *
 * Usage:
 *   vd100-ps-ma-client <xclbin> [price_file]
 *
 *   price_file: optional, one int32 price per line.
 *               Must be a multiple of BLOCK_SIZE (56) samples.
 *               Defaults to built-in golden test vector if omitted.
 *
 * Expected output (golden test vector — matches golden.txt exactly):
 *   Block 1: fast_ma=5000  slow_ma=5000  HOLD
 *   Block 2: fast_ma=4990  slow_ma=4990  HOLD
 *   Block 3: fast_ma=5051  slow_ma=5002  BUY   <- crossover triggered
 *   Block 4: fast_ma=5600  slow_ma=5600  HOLD
 */

//#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
//#include <xrt/xrt_bo.h>
#include <xrt/xrt_graph.h>
//#include <xrt/experimental/xrt_xclbin.h>  // XRT 2025.2 Yocto: xrt_xclbin.h is in experimental/
//#include <xrt/experimental/xrt_aie.h>
#include <xrt/xrt_aie.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/ioctl.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include <dirent.h>

// --- AIE kernel constants (must match kernels.cc) ----------------------------
static constexpr int BLOCK_SIZE  = 56;   // int32 samples per AIE iteration (56x4=224B, multiple of 32)
static constexpr int OUTPUT_VALS = 3;    // fast_ma, slow_ma, signal per block
static constexpr int SIGNAL_BUY  =  1;
static constexpr int SIGNAL_SELL = -1;

// --- Helpers -----------------------------------------------------------------
static const char* signal_str(int32_t s) {
    if (s == SIGNAL_BUY)  return "BUY ";
    if (s == SIGNAL_SELL) return "SELL";
    return "HOLD";
}

/** 4-block golden test vector -- must match golden.txt exactly */
static std::vector<int32_t> make_golden_input() {
    std::vector<int32_t> v;
    v.reserve(BLOCK_SIZE * 4);
    for (int i = 0; i < 56; ++i) v.push_back(5000);  // Block 1: 56 x 5000
    for (int i = 0; i < 56; ++i) v.push_back(4990);  // Block 2: 56 x 4990
    for (int i = 0; i < 55; ++i) v.push_back(4990);  // Block 3: 55 x 4990
    v.push_back(5600);                                 // Block 3: 1 spike -> BUY
    for (int i = 0; i < 56; ++i) v.push_back(5600);  // Block 4: 56 x 5600
    return v;
}

/** Load prices from file -- one int32 per line */
static std::vector<int32_t> load_price_file(const char* path) {
    std::vector<int32_t> v;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "ERROR: cannot open price file: %s\n", path);
        return {};
    }
    int32_t x;
    while (f >> x) v.push_back(x);
    if (v.size() % BLOCK_SIZE != 0) {
        fprintf(stderr, "ERROR: price file has %zu samples -- must be multiple of %d\n",
                v.size(), BLOCK_SIZE);
        return {};
    }
    return v;
}

// --- Main --------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <binary_container_1.xclbin> [price_file]\n"
                "  price_file: optional, one int32/line, must be multiple of %d samples\n",
                argv[0], BLOCK_SIZE);
        return 1;
    }

    const char* xclbin_path = argv[1];
    const char* price_file  = (argc > 2) ? argv[2] : nullptr;

    // 1. Load prices ----------------------------------------------------------
    std::vector<int32_t> prices;
    if (price_file) {
        prices = load_price_file(price_file);
        if (prices.empty()) return 1;
        printf("[INFO] Loaded %zu prices (%zu blocks) from %s\n",
               prices.size(), prices.size() / BLOCK_SIZE, price_file);
    } else {
        prices = make_golden_input();
        printf("[INFO] Using built-in golden test vector (%zu blocks)\n",
               prices.size() / BLOCK_SIZE);
    }

    const int num_blocks    = static_cast<int>(prices.size() / BLOCK_SIZE);
    const int input_samples = num_blocks * BLOCK_SIZE;   // int32 count for mm2s
    const int output_vals   = num_blocks * OUTPUT_VALS;  // int32 count for s2mm

    // 2. Open device ----------------------------------------------------------
    printf("[INFO] Opening device 0\n");
    xrt::device device(0);

    //  auto reset_done = xrtResetAIEArray(device);
    //  if (reset_done == -1){
    //      printf("[Beamformer] AIE reset FAILS \n");
    //      //printf ("[Beamformer] AIE reset FAILS\n");;
    //      printf("[Beamformer] Exiting the application as it cannot be run\n");
    //      //printf ("[Beamformer] Exiting the application as it cannot be run" \n");
    // }
    //  else if(reset_done == 0) {
    //     printf("[Beamformer] AIE reset done successfully\n");
    //      //log_plnx << "[Beamformer] AIE reset done successfully" << std::endl;
    //  }

    // 3. Load xclbin -- XRT 2025.2: register_xclbin() + hw_context ----------
    printf("[INFO] Loading xclbin: %s\n", xclbin_path);
    xrt::xclbin xclbin_obj{std::string{xclbin_path}};  // brace-init: avoids vexing parse; name avoids namespace xclbin conflict
    auto uuid = device.register_xclbin(xclbin_obj);
    //auto uuid = device.load_xclbin(std::string{xclbin_path});

    //auto uuid = xclbin_obj.get_uuid();
    printf("[INFO] xclbin registered -- UUID: %s\n", uuid.to_string().c_str());

   // int aie_fd = open("/dev/aie0", O_RDWR);
   // printf("[INFO] /dev/aie0 fd=%d\n", aie_fd);

    // 4. Create hw_context -- required for kernel creation in XRT 2025.2 -----
    xrt::hw_context ctx(device, uuid);
    printf("[INFO] hw_context created\n");


    // struct aie_column_args { uint32_t start_col; uint32_t num_cols; uint8_t enable; };
    // #define AIE_SET_COLUMN_CLOCK_IOCTL _IOW('A', 0x1b, struct aie_column_args)
    //
    // char path[64], link[256];
    // for (int f = 3; f < 32; f++) {
    //     snprintf(path, sizeof(path), "/proc/self/fd/%d", f);
    //     ssize_t n = readlink(path, link, sizeof(link)-1);
    //     if (n > 0 && (link[n]=0, strstr(link, "aiepart"))) {
    //         int pfd = dup(f);
    //         struct aie_column_args args = {0, 17, 1};
    //         int r = ioctl(pfd, AIE_SET_COLUMN_CLOCK_IOCTL, &args);
    //         printf("[INFO] SET_COLUMN_CLOCK: fd=%d ret=%d\n", pfd, r);
    //         close(pfd);
    //         break;
    //     }
    // }
    //
    //
    // 6. Get HLS kernel handles via hw_context (XRT 2025.2 preferred) --------
    //    mm2s(ap_int<32>* mem, stream& s, int size) -> mem = arg[0], group_id(0)
    //    s2mm(ap_int<32>* mem, stream& s, int size) -> mem = arg[0], group_id(0)
    //
    //    Kernel names confirmed via xclbinutil --info:
    //      Kernels: s2mm, mm2s  (no _1 suffix in xclbin metadata)
    //      Signature: f(void* mem, stream& s, unsigned int size)
    //    Address editor /VitisRegion/mm2s_1 is the instance path -- irrelevant here.
    printf("[INFO] Instantiating Kernels\n");
    //xrt::kernel mm2s_k(device, uuid, "mm2s:{mm2s_1}");
    //xrt::kernel s2mm_k(device, uuid, "s2mm:{s2mm_1}");
	auto s2mm_k = xrt::kernel(ctx, "s2mm:{s2mm_1}");
	auto mm2s_k = xrt::kernel(ctx, "mm2s:{mm2s_1}");
	
    printf("[INFO] Kernels acquired: mm2s group_id=%d  s2mm group_id=%d\n",  mm2s_k.group_id(0), s2mm_k.group_id(0));

    // 7. Allocate BOs ---------------------------------------------------------
    printf("[INFO] Allocating BOs -- input: %d B  output: %d B\n", input_samples * 4, output_vals * 4);

    // output memory for PL kernel
    auto out_bo = xrt::bo(ctx, static_cast<size_t>(output_vals) * sizeof(int32_t),static_cast<xrt::bo::flags>(0), s2mm_k.group_id(0));
    //xrt::bo out_bo(device, static_cast<size_t>(output_vals) * sizeof(int32_t), s2mm_k.group_id(0));

    {
        auto* ptr = out_bo.map<int32_t*>();
        memset(ptr, 0, output_vals * sizeof(int32_t));
    }
	
	auto in_bo = xrt::bo(ctx, static_cast<size_t>(input_samples) * sizeof(int32_t),static_cast<xrt::bo::flags>(0), mm2s_k.group_id(0));
    //auto host_in=in_bo.map<int*>();

    //xrt::bo in_bo(device, static_cast<size_t>(input_samples) * sizeof(int32_t),mm2s_k.group_id(0));

    // 8. Fill input BO and zero output BO -------------------------------------
    {
        auto* ptr = in_bo.map<int32_t*>();
        memcpy(ptr, prices.data(), prices.size() * sizeof(int32_t));
    }

    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("[INFO] Input synced to device\n");

    // 5. Get AIE graph handle -------------------------------------------------
    //    xrt::graph(device, uuid, name) -- unchanged in XRT 2025.2
    printf("[INFO] Acquiring AIE graph: mygraph\n");
    xrt::graph graph(ctx, "mygraph");

    // 9. Reset + run graph ----------------------------------------------------
    graph.reset();
    printf("[INFO] Graph reset\n");

    printf("[INFO] Run Graph -- %d iteration(s)\n", num_blocks);
    graph.run(-1);
    printf("[INFO] Graph running -- %d iteration(s)\n", num_blocks);


    // 10. Launch HLS DMA kernels ----------------------------------------------
    //     ORDERING: s2mm FIRST -- must be ready to sink from AIE before mm2s fires.
    //     If mm2s fires first, AIE output stream backs up and stalls the graph.
    //
    //     arg[0] = BO  |  arg[1] = stream (skipped, connectivity handles it)  |  arg[2] = size

    // size is unsigned int per xclbinutil signature — cast explicitly
    xrt::run s2mm_run = s2mm_k(out_bo, nullptr, static_cast<unsigned int>(output_vals) * sizeof(int32_t));
   // s2mm_run.set_arg(0, out_bo);
   // s2mm_run.set_arg(2, static_cast<unsigned int>(output_vals));
    //s2mm_run.start();
    printf("[INFO] s2mm started (sink ready)\n");

    xrt::run mm2s_run = mm2s_k(in_bo,nullptr,static_cast<unsigned int>(input_samples) * sizeof(int32_t)) ;
   // mm2s_run.set_arg(0, in_bo);
   // mm2s_run.set_arg(2, static_cast<unsigned int>(input_samples) * sizeof(int32_t));
    ///mm2s_run.start();
    printf("[INFO] mm2s started (source firing)\n");


    printf("[INFO] Waiting for graph ...\n");
    graph.wait(2000);
	
	printf("[INFO] Waiting for s2mm ...\n");
    s2mm_run.wait(2000);
    printf("[INFO] s2mm done\n");
	
    // 11. Wait for completion -------------------------------------------------
    printf("[INFO] Waiting for mm2s ...\n");
    mm2s_run.wait(2000);
    printf("[INFO] mm2s done\n");

    printf("[INFO] Graph done\n");
    //graph.end();
    // 12. Sync output from device ---------------------------------------------
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // 13. Print results -------------------------------------------------------
    const int32_t* out = out_bo.map<const int32_t*>();

    printf("\n");
    printf("+---------+----------+----------+--------+\n");
    printf("|  Block  | fast_ma  | slow_ma  | Signal |\n");
    printf("+---------+----------+----------+--------+\n");

    for (int b = 0; b < num_blocks; ++b) {
        int32_t fast_ma = out[b * OUTPUT_VALS + 0];
        int32_t slow_ma = out[b * OUTPUT_VALS + 1];
        int32_t signal  = out[b * OUTPUT_VALS + 2];
        printf("|  %5d  |  %6d  |  %6d  |  %s  |\n",
               b + 1, fast_ma, slow_ma, signal_str(signal));
    }

    printf("+---------+----------+----------+--------+\n");
    printf("\n[DONE] %d block(s) processed via AIE-ML v2 (XCVE2302)\n", num_blocks);

    return 0;
}
