"""
source:
https://github.com/open-neuromorphic/open-neuromorphic.github.io/tree/main/content/english/blog/spiking-neural-network-framework-benchmarking
"""

import torch
import numpy as np
from utils import benchmark_framework, allplatformscallable


def ship():
    from SHIP import network, inputN,wireS,lifN,SurrGradTrainer,__version__

    benchmark_title = f"SHIP<br>v{__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        snn = network()
        snn.add(inputN,"I",N=n_neurons,device = device)
        snn.add(lifN,"L",N=n_neurons,is_output = True,device = device)
        snn.add(wireS,"IL",source = "I",target="L")
        snn.init(dt = 1e-3, nts = n_steps, batch_size = batch_size)
        trn = SurrGradTrainer(snn)
        trn.init()
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        return dict(model=snn, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        model.run(input_static)
        bench_dict["output"] = model.output
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

allplatforms = allplatformscallable()
allplatforms.append(ship)

batch_size = 10
n_steps = 500
n_layers = 1  # doesn't do anything at the moment
device = "cpu"

data = []
for benchmark in allplatforms:
    for n_neurons in [32,64,128,256,512,1024,2048,4096,8192,16384]:
        prepare_fn, forward_fn, backward_fn, bench_desc = benchmark()
        print("Benchmarking", bench_desc, "with n_neurons =", n_neurons)
        try:
            forward_times, backward_times = benchmark_framework(
                prepare_fn=prepare_fn,
                forward_fn=forward_fn,
                backward_fn=backward_fn,
                benchmark_desc=bench_desc,
                n_neurons=n_neurons,
                n_layers=n_layers,
                n_steps=n_steps,
                batch_size=batch_size,
                device=device,
            )
            data.append(
                [
                    bench_desc,
                    np.array(forward_times).mean(),
                    np.array(backward_times).mean(),
                    n_neurons,
                ]
            )
        except:
            print("issue with "+bench_desc)
            
import pandas as pd

df = pd.DataFrame(data, columns=["framework", "forward", "backward", "neurons"])
df = df.melt(
    id_vars=["framework", "neurons"],
    value_vars=["forward", "backward"],
    var_name="pass",
    value_name="time [s]",
)
df.to_csv("data.csv")