import pandas as pd
from unum import Unum
from data.sys_data import sys_data

s = Unum.unit('s')
Hz = Unum.unit('Hz', s)
GHz = Unum.unit('GHz', 10**9 * Hz)
B = Unum.unit('B')  # byte
GB = Unum.unit('GB', 10**9 * B)
FLOPS = Unum.unit('FLOPS')
GFLOPS = Unum.unit('GFLOPS', 10**9 * FLOPS)
FLOPSS = Unum.unit('FLOPS/s', FLOPS/s, FLOPS*Hz)
GFLOPSS = Unum.unit('GFLOPS/s', FLOPS*GHz)
N = Unum.unit('N')


class data:
    def parse_data(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        return data

    def get_test_data() -> pd.DataFrame:
        data = pd.read_csv('test_data.txt')
        return data

    def effective_bandwidth(data_transfered: float, time: float,
                            rw: bool) -> float:
        if rw:
            return data_transfered / time * 2
        return data_transfered / time

    def flopss(total_flops: float, time: float) -> float:
        """
        Floating point operations per second.
        """
        return total_flops / time

    def arithmetic_intensity(flops: float, data_transfered: float) -> float:
        """
        Arithmetic intensity is the ratio of flops to data transfered.
        A high AI value means we’re doing a lot of computation per load,
        while a low AI means the algorithm is often waiting on data
        to be loaded before it can do anything (memory bound).
        """
        return flops / data_transfered

    def operational_intensity(flops: float, data_transfered: float) -> float:
        return flops / data_transfered

    def attainable_performance(memory_peak: float,
                               operational_intensity: float,
                               performance_peak: float) -> float:
        return min(memory_peak * operational_intensity, performance_peak)

    def performance_peak(core_count: int, clock_speed: float,
                         flops_per_cycle: int) -> float:  # GFLOPSS
        """
        The peak performance of a processor is the maximum number of
        floating point operations it can perform in a second.
        - get the core count: lscpu
            - Threads per core X cores per socket X sockets
        - get the clock speed (GHz): lscpu | grep MHz
        - flops_per_cycle: check wikipedia
           - https://en.wikipedia.org/wiki/Floating_point_operations_per_second

        """
        return core_count * clock_speed * flops_per_cycle

    def memory_peak() -> float:  # GB/s
        return sys_data.peak_theoretical_bandwidth

    def roofing_ridge(performance_peak: float, memory_peak: float) -> float:
        return performance_peak / memory_peak

    def amdahl_law(parallelizable_fraction: float, core_count: int) -> float:
        """
        Amdahl’s Law is a formula that describes the maximum speedup
        that can be achieved by parallelizing a computation.

        It determined Strong scaling: the scalability of a program should be
        measured by increasing the resources and keeping the
        problem size
        """
        return 1 / ((1 - parallelizable_fraction) +
                    parallelizable_fraction / core_count)

    def gustafson_law(alpha: float,
                      core_count: int) -> float:
        """
        Gustafson’s Law is a formula that describes the maximum
        speedup that can be achieved by parallelizing a computation.

        It determines Weak scaling: the scalability of a program
        should be measured by increasing the resources and the
        problem size.

        - alpha is the fraction of the execution time of the
        parallelizable portion of the task on a single processor.
        """
        return core_count + (1 - core_count) * alpha
