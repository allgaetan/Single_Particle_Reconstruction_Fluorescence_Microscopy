from common_image_processing_methods.others import *
from reconstruction import reconstruct
import torch
from torch.profiler import profile, ProfilerActivity

def profile(params_data_gen, params_learn_setup):
    """
    Code inspired by https://huggingface.co/blog/torch-profiler
    Permet de générer des traces de profilage pour la reconstruction avec HPS
    - params_data_gen: paramètres de génération de données
    - params_learn_setup: paramètres de l'apprentissage
    """
    trace_dir = params_learn_setup["save_fold"] + "/profiling"

    def step():
        with torch.profiler.record_function("reconstruct"):
            return reconstruct(params_data_gen, params_learn_setup)

    os.makedirs(trace_dir, exist_ok=True)
    compile_tag = "compile"
    warmup_tag = "warm" 
    tag = f"{warmup_tag}_{compile_tag}"
    table_path = os.path.join(trace_dir, f"{tag}.txt")
    trace_path = os.path.join(trace_dir, f"{tag}.json")

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        record_shapes=False, 
        profile_memory=False, 
        with_stack=False,
    ) as prof:
        for _ in range(5):
            step()
            prof.step()

    torch.cuda.synchronize()

    print(f"saving traces ... {trace_path}")
    prof.export_chrome_trace(trace_path)
    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))