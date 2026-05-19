import time
import torch
import psutil


def benchmark_generation(model, tokenizer, prompt):

    start_time = time.time()

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False
    )

    end_time = time.time()

    latency = end_time - start_time

    generated_tokens = outputs[0].shape[0]

    throughput = generated_tokens / latency

    gpu_memory = torch.cuda.memory_allocated()

    gpu_memory_gb = gpu_memory / (1024 ** 3)

    cpu_memory = psutil.virtual_memory().used

    cpu_memory_gb = cpu_memory / (1024 ** 3)

    return {
        "latency_seconds": latency,
        "throughput_tokens_per_second": throughput,
        "gpu_memory_gb": gpu_memory_gb,
        "cpu_memory_gb": cpu_memory_gb
    }