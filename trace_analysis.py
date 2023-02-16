# %%
from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir="output/traces/1")

# %% Temporal breakdown
temporal_breakdown_df = analyzer.get_temporal_breakdown()
temporal_breakdown_df

# %% Idle time breakdown
idle_time_df = analyzer.get_idle_time_breakdown()
idle_time_df

# %% Kernel breakdown
kernel_breakdown_df = analyzer.get_gpu_kernel_breakdown()
kernel_breakdown_df

# %% Communication computation overlap
comm_comp_overlap_df = analyzer.get_comm_comp_overlap()
comm_comp_overlap_df

# %% Memory bandwidth time series
memory_bw_series = analyzer.get_memory_bw_time_series()
memory_bw_series

# %% Memory bandwidth summary
memory_bw_summary = analyzer.get_memory_bw_summary()
memory_bw_summary

# %% Queue length time series
ql_series = analyzer.get_queue_length_time_series()
ql_series

# %% Queue length summary
ql_summary = analyzer.get_queue_length_summary()
ql_summary

# %% CUDA kernel launch statistics
cuda_kernel_launch_stats = analyzer.get_cuda_kernel_launch_stats()
cuda_kernel_launch_stats

# %% Frequent CUDA kernel sequences
patterns = {}
for operator in ["aten::linear", "aten::conv"]:
    patterns[operator] = analyzer.get_frequent_cuda_kernel_sequences(
        operator_name=operator, output_dir="/output/traces/", vizualize=True
    )
patterns
