class cluster_data:
    """
    This class contains the system data that is used to calculate the
    theoretical peak bandwidth of the cluster.
    """

    megabyte = 1024 * 1024
    # sudo dmidecode --type 17 | grep "Configured Memory Speed"
    memory_clock_rate = 3200 * megabyte  # BT/s
    # sudo lshw -C display | grep width
    bus_width = 64  # bits
    transfer_rate = 2  # 2 for is DDR

    peak_theoretical_bandwidth = \
        memory_clock_rate * bus_width * transfer_rate / 10e9  # GB/s

    core_count = 64
    clock_speed = 2.3 # GHz
    flops_per_cycle = 8  # 8 for AVX
