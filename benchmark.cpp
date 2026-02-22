#include "benchmark.h"

#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <iostream>
#include <algorithm>

#include <omp.h>
#include <cuda_runtime_api.h>

#include "gpu_reductions.h"
#include "gpu_memory_management.h"
#include "intermediate_image.h"

namespace {

std::vector<double> make_image(std::size_t n, std::uint32_t seed) {
    std::vector<double> v(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 255.0);
    for (std::size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

double max_serial(const double* data, std::size_t n) {
    if (n == 0) return 0.0;
    double m = data[0];
    for (std::size_t i = 1; i < n; ++i) m = (data[i] > m) ? data[i] : m;
    return m;
}

double max_omp(const double* data, std::size_t n) {
    if (n == 0) return 0.0;
    double m = data[0];
    #pragma omp parallel for reduction(max:m)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        m = (data[static_cast<std::size_t>(i)] > m) ? data[static_cast<std::size_t>(i)] : m;
    }
    return m;
}

static void bench_serial(benchmark::State& state) {
    const std::uint32_t h = static_cast<std::uint32_t>(state.range(0));
    const std::uint32_t w = static_cast<std::uint32_t>(state.range(1));
    const std::size_t n = static_cast<std::size_t>(h) * w;

    state.PauseTiming();
    auto img = make_image(n, 123u);
    state.ResumeTiming();

    for (auto _ : state) {
        const double r = max_serial(img.data(), img.size());
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(n));
}

static void bench_omp(benchmark::State& state) {
    const std::uint32_t h = static_cast<std::uint32_t>(state.range(0));
    const std::uint32_t w = static_cast<std::uint32_t>(state.range(1));
    const std::size_t n = static_cast<std::size_t>(h) * w;

    state.PauseTiming();
    auto img = make_image(n, 123u);
    state.ResumeTiming();

    for (auto _ : state) {
        const double r = max_omp(img.data(), img.size());
        benchmark::DoNotOptimize(r);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(n));
}

static void bench_gpu(benchmark::State& state) {
    const std::uint32_t h = static_cast<std::uint32_t>(state.range(0));
    const std::uint32_t w = static_cast<std::uint32_t>(state.range(1));
    const std::size_t n = static_cast<std::size_t>(h) * w;

    state.PauseTiming();

    cudaFree(0);

    IntermediateImage img;
    img.height = h;
    img.width  = w;
    img.pixels = make_image(n, 123u);

    void* d_img = nullptr;
    allocate_device_memory(img, &d_img);
    copy_data_to_device(img, &d_img);

    (void)get_max_value(&d_img, h, w);

    state.ResumeTiming();

    for (auto _ : state) {
        const double r = get_max_value(&d_img, h, w);
        benchmark::DoNotOptimize(r);
    }

    state.PauseTiming();
    free_device_memory(&d_img);
    state.ResumeTiming();

    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(n));
}

void register_all() {
    const int T = omp_get_max_threads();
    omp_set_num_threads(T);

    const std::vector<std::uint32_t> serial_sizes = {256, 512, 1024, 2048, 4096};
    const std::vector<std::uint32_t> par_sizes    = {256, 512, 1024, 2048, 4096, 8192};

    for (auto s : serial_sizes) {
        benchmark::RegisterBenchmark("get_max_serial", bench_serial)
            ->Args({s, s})
            ->Unit(benchmark::kMillisecond);
    }

    for (auto s : par_sizes) {
        benchmark::RegisterBenchmark(("get_max_omp_T" + std::to_string(T)).c_str(), bench_omp)
            ->Args({s, s})
            ->Unit(benchmark::kMillisecond);
    }

    for (auto s : par_sizes) {
        benchmark::RegisterBenchmark("get_max_gpu", bench_gpu)
            ->Args({s, s})
            ->Unit(benchmark::kMillisecond);
    }
}

}

int main(int argc, char** argv) {
    register_all();
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
