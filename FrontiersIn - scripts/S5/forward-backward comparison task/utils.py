"""
source:
https://github.com/open-neuromorphic/open-neuromorphic.github.io/tree/main/content/english/blog/spiking-neural-network-framework-benchmarking
"""

from typing import Callable, Optional, List, Tuple
import warnings

from time import time


def timeit(
    callable: Callable,
    min_runs: int = 3,
    max_runs: int = 5000,
    min_time: float = 2.0,
    warmup_calls: int = 1,
) -> List[float]:

    # - Warmup
    for _ in range(warmup_calls):
        callable()

    # - Take at least min_time seconds, at least min_runs runs
    exec_count = 0
    t_total = 0.0
    collected_times = []
    while ((t_total <= min_time) or (exec_count < min_runs)) and (
        exec_count < max_runs
    ):
        # - Time a single run
        t_start = time()
        callable()
        collected_times.append(time() - t_start)

        exec_count += 1
        t_total = sum(collected_times)

    return collected_times


def benchmark_framework(
    prepare_fn: Callable,
    forward_fn: Callable,
    backward_fn: Callable,
    benchmark_desc: Optional[str] = None,
    n_neurons: int = 512,
    n_layers: int = 4,
    n_steps: int = 500,
    batch_size: int = 10,
    device: str = "cpu",
) -> Tuple[List, List, List]:
    forward_times = []
    backward_times = []

    try:
        # - Prepare benchmark
        bench_dict = prepare_fn(
            batch_size=batch_size, 
            n_steps=n_steps, 
            n_neurons=n_neurons,
            n_layers=n_layers, 
            device=device
        )

        # - Forward pass
        forward_times.append(timeit(lambda: forward_fn(bench_dict)))
        bench_dict = forward_fn(bench_dict)
        assert bench_dict["output"].shape == bench_dict["input"].shape

        # - Backward pass
        backward_times.append(timeit(lambda: backward_fn(bench_dict)))

    except Exception as e:
        # - Fail nicely with a warning if a benchmark dies
        warnings.warn(
        f"Benchmark {benchmark_desc} failed with error {str(e)}."
        )

        # - No results for this run
        forward_times.append([])
        backward_times.append([])

    # - Build a description of the benchmark
    benchmark_desc = f"{benchmark_desc}; " if benchmark_desc is not None else ""
    benchmark_desc = f"{benchmark_desc}B = {batch_size}, T = {n_steps}"

    # - Return benchmark results
    return forward_times, backward_times



import torch
from torch import nn


def rockpool_torch():
    from rockpool.nn.modules import LIFTorch, LinearTorch
    from rockpool.nn.combinators import Sequential
    import rockpool

    benchmark_title = f"Rockpool<br>v{rockpool.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        model = Sequential(
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFTorch(n_neurons),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        output = model(input_static)[0]
        bench_dict["output"] = output
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def rockpool_exodus():
    from rockpool.nn.modules import LIFExodus, LinearTorch
    from rockpool.nn.combinators import Sequential
    import rockpool

    benchmark_title = f"Rockpool EXODUS<br>v{rockpool.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        model = Sequential(
            LinearTorch(shape=(n_neurons, n_neurons)),
            LIFExodus(n_neurons),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        output = model(input_static)[0]
        bench_dict["output"] = output
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def sinabs():
    from sinabs.layers import LIF
    import sinabs

    benchmark_title = f"Sinabs<br>v{sinabs.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        model = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            LIF(tau_mem=torch.tensor(10.0)),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        sinabs.reset_states(model)
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def sinabs_exodus():
    from sinabs.exodus.layers import LIF
    import sinabs

    benchmark_title = f"Sinabs EXODUS<br>v{sinabs.exodus.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        model = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            LIF(tau_mem=torch.tensor(10.0)),
        ).to(device)
        input_static = torch.randn(batch_size, n_steps, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        sinabs.reset_states(model)
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def norse():
    from norse.torch.module.lif import LIF
    from norse.torch import SequentialState
    import norse

    benchmark_title = f"Norse<br>v{norse.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        model = SequentialState(
            nn.Linear(n_neurons, n_neurons),
            LIF(),
        )
        # model = torch.compile(model, mode="max-autotune")
        model = model.to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        # output.sum().backward() # JIT compile everything
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)[0]
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def snntorch():
    import snntorch

    benchmark_title = f"snnTorch<br>v{snntorch.__version__}"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            def __init__(self, beta: float = 0.95):
                super().__init__()
                self.fc = nn.Linear(n_neurons, n_neurons)
                self.lif = snntorch.Leaky(beta=beta)
                self.mem = self.lif.init_leaky()

            def forward(self, x):
                output = []
                mem = self.mem
                for inp in x:
                    cur = self.fc(inp)
                    spk, mem = self.lif(cur, mem)
                    output.append(spk)
                return torch.stack(output)

        model = Model()
        # model = torch.compile(model, mode="max-autotune")
        model = model.to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

# mix of https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/basic_concept.html#step-mode
# and https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/rsnn_sequential_fmnist.py
def spikingjelly():
    from spikingjelly.activation_based import neuron, surrogate, functional, layer

    benchmark_title = f"SpikingJelly PyTorch<br>v0.0.0.0.15"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            def __init__(self, tau=5.0):
                super().__init__()
                self.model = nn.Sequential(
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m'),
                )

            def forward(self, x):
                functional.reset_net(self.model)
                return self.model(x)

        model = Model().to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def spikingjelly_cupy():
    from spikingjelly.activation_based import neuron, surrogate, functional, layer

    benchmark_title = f"SpikingJelly CuPy<br>v0.0.0.0.15"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        class Model(nn.Module):
            def __init__(self, tau=5.0):
                super().__init__()
                self.model = nn.Sequential(
                    layer.Linear(n_neurons, n_neurons),
                    neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m'),
                )
                functional.set_backend(self.model, backend='cupy')

            def forward(self, x):
                functional.reset_net(self.model)
                return self.model(x)

        model = Model().to(device)
        input_static = torch.randn(n_steps, batch_size, n_neurons).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title

def lava():
    import lava.lib.dl.slayer as slayer

    benchmark_title = f"Lava DL<br>v0.4.0.dev0"

    def prepare_fn(batch_size, n_steps, n_neurons, n_layers, device):
        neuron_params = {
                        'threshold'     : 0.1,
                        'current_decay' : 1,
                        'voltage_decay' : 0.1,
                        'requires_grad' : True,     
                    }
        # slayer.block automatically add quantization.
        # They can be disabled by setting pre_hook_fx=None
        model = slayer.block.cuba.Dense(neuron_params, n_neurons, n_neurons, pre_hook_fx=None).to(device)
        input_static = torch.randn(batch_size, n_neurons, n_steps).to(device)
        with torch.no_grad():
            model(input_static)
        return dict(model=model, input=input_static, n_neurons=n_neurons)

    def forward_fn(bench_dict):
        model, input_static = bench_dict["model"], bench_dict["input"]
        bench_dict["output"] = model(input_static)
        return bench_dict

    def backward_fn(bench_dict):
        output = bench_dict["output"]
        loss = output.sum()
        loss.backward(retain_graph=True)

    return prepare_fn, forward_fn, backward_fn, benchmark_title


def allplatformscallable():
    return [lava, rockpool_torch,  sinabs, snntorch,]# sinabs_exodus,  norse,  spikingjelly_cupy, rockpool_exodus