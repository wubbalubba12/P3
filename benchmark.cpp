#include "benchmark.h"

#include <cstdint>
#include <vector>
#include <iostream>

static void dummy_benchmark(benchmark::State& state) {
	const auto n = state.range(0);
	
	for (auto _ : state) {
		state.PauseTiming();

		state.ResumeTiming();
		//benchmark::DoNotOptimize(autStat.print());
	}	
}

int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv))
		return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}


// DUMMY
BENCHMARK(dummy_benchmark)->Unit(benchmark::kMillisecond)->Arg(100);