"""SHR-PAPER-01-A: source sharing ablation, with unchanged production decoder.

The pathwise bank contains an independently computed/aggregated slot for each
full path. Both variants use ONE packed sparse aggregation call. The production
decoder visits every (target group, source key) once; audit enforces its bijection
to full paths before using the consumable bank. No training code is modified.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
import torch
import e3nn
from shr_operator.primitives import regular_solid_harmonics

core = importlib.import_module('shr_operator.shr')


def key(b):
    return b.input_degree, b.source_degree, b.intermediate_degree


class PathwiseBank:
    def __init__(self, slots):
        self.slots = slots
        self.used = Counter()

    def __getitem__(self, k):
        i = self.used[k]
        self.used[k] += 1
        return self.slots[k][i]


class AblationSHR(core.SparseHarmonicRecoupling):
    pathwise = False

    def build_local_moments(self, features, positions, edge_index, alpha,
                            batch=None, *, centered_positions=None,
                            node_geometry=None, num_target_nodes=None):
        if not self.pathwise:
            return super().build_local_moments(
                features, positions, edge_index, alpha, batch,
                centered_positions=centered_positions, node_geometry=node_geometry,
                num_target_nodes=num_target_nodes)
        alpha = self._validate(features, positions, edge_index, alpha, batch)
        pos = centered_positions if centered_positions is not None else core.center_positions(positions, batch)
        geom = self._geometry(pos, node_geometry)
        nt = pos.shape[0] if num_target_nodes is None else num_target_nodes
        if not 0 <= nt <= pos.shape[0]:
            raise ValueError('invalid target count')
        # Sorting matches the shared builder's key order, but keeps duplicates.
        keys = sorted(key(b) for b in self.plan.branches)
        parts, widths = [], []
        for l, a, f in keys:
            src = core.component_tensor_product(features[l], geom[a], l, a, f)
            parts.append(src.reshape(pos.shape[0], -1))
            widths.append(src.shape[1] * src.shape[2])
        stacked = torch.cat(parts, dim=-1)
        aggregated = core.weighted_aggregate(stacked, alpha, edge_index, nt)
        slots, offset = defaultdict(list), 0
        for k, width in zip(keys, widths):
            slots[k].append(aggregated[:, offset:offset+width].reshape(nt, 2*k[2]+1, self.in_channels))
            offset += width
        return PathwiseBank(slots)


def audit(model, h, pos, edges, alpha, geom):
    consumers = Counter((b.output_degree, b.intermediate_degree, b.target_degree, key(b))
                        for b in model.plan.branches)
    assert all(v == 1 for v in consumers.values()), 'decoder merges full paths: revise baseline'
    result = {}
    for pathwise in (False, True):
        model.pathwise = pathwise
        counts = dict(tp=0, agg=0, width=0)
        original_tp, original_agg = core.component_tensor_product, core.weighted_aggregate
        def tp(*args, **kwargs):
            counts['tp'] += 1
            return original_tp(*args, **kwargs)
        def agg(*args, **kwargs):
            counts['agg'] += 1
            counts['width'] = args[0].shape[1]
            return original_agg(*args, **kwargs)
        # Plain wrappers do not retain tensor call histories like MagicMock.
        with torch.no_grad(), patch.object(core, 'component_tensor_product', tp), patch.object(core, 'weighted_aggregate', agg):
            moments = model.build_local_moments(h, pos, edges, alpha, centered_positions=pos, node_geometry=geom)
            source_calls, aggregate_calls = counts['tp'], counts['agg']
            packed_width = counts['width']
            model.decode_moments(moments, pos, centered_positions=pos, node_geometry=geom)
            if pathwise:
                assert moments.used == Counter(key(b) for b in model.plan.branches)
                for slots in moments.slots.values():
                    assert len({s.data_ptr() for s in slots}) == len(slots)
            result['pathwise' if pathwise else 'shared'] = dict(
                source_tp_calls=source_calls, aggregation_calls=aggregate_calls,
                packed_width=packed_width, target_tp_calls=counts['tp']-source_calls)
    assert result['shared']['source_tp_calls'] == len({key(b) for b in model.plan.branches})
    assert result['pathwise']['source_tp_calls'] == len(model.plan.branches)
    assert result['shared']['target_tp_calls'] == result['pathwise']['target_tp_calls']
    assert result['shared']['aggregation_calls'] == result['pathwise']['aggregation_calls'] == 1
    return result


def compare(xs, ys, atol, rtol):
    maximum, total, count, refmax = 0., 0., 0, 0.
    for x, y in zip(xs, ys):
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
        d = (x-y).abs()
        maximum = max(maximum, d.max().item())
        total += d.double().sum().item()
        count += d.numel()
        refmax = max(refmax, y.abs().max().item())
    return dict(max_abs=maximum, mean_abs=total/count, reference_max_abs=refmax,
                atol=atol, rtol=rtol, passed=True)


def inputs(cfg, dtype, device, n=None, c=None):
    # CPU RNG guarantees the same base graph/features for every G and device.
    rng = torch.Generator().manual_seed(cfg['seed'])
    n, c = n or cfg['nodes'], c or cfg['channels']
    pos = torch.randn(n, 3, generator=rng, dtype=torch.double)*.2
    pos -= pos.mean(0)
    h = {l: torch.randn(n, 2*l+1, c, generator=rng, dtype=torch.double).to(device=device, dtype=dtype).requires_grad_()
         for l in range(cfg['L']+1)}
    src = torch.randint(n, (n*cfg['neighbors'],), generator=rng)
    dst = torch.arange(n).repeat_interleave(cfg['neighbors'])
    edges = torch.stack((src, dst)).to(device)
    alpha = torch.randn(n, cfg['neighbors'], generator=rng, dtype=torch.double).softmax(1).flatten().to(device=device, dtype=dtype).requires_grad_()
    return h, pos.to(device=device, dtype=dtype), edges, alpha


def validate(model, h, pos, edges, alpha, G):
    dtype = pos.dtype
    # Geometry leaves check both source and target geometry derivatives, without
    # including common harmonic construction in timed operator measurements.
    geom = {l: g.detach().requires_grad_() for l, g in regular_solid_harmonics(pos, G).items()}
    names = [f'feature_{l}' for l in h] + ['alpha'] + [f'geometry_{l}' for l in geom] + [f'weight_{k}' for k in model.path_weights]
    leaves = list(h.values())+[alpha]+list(geom.values())+list(model.parameters())
    records = []
    cot = None
    for pathwise in (False, True):
        model.pathwise = pathwise
        out = model(h, pos, edges, alpha, centered_positions=pos, node_geometry=geom)
        if cot is None:
            rng = torch.Generator(device=pos.device).manual_seed(992)
            cot = [torch.randn(v.shape, dtype=dtype, device=pos.device, generator=rng)/v.numel()**.5 for v in out.values()]
        grads = torch.autograd.grad(tuple(out.values()), leaves, grad_outputs=cot)
        records.append(([v.detach() for v in out.values()], grads))
    atol, rtol = (1e-10, 1e-9) if dtype == torch.double else (3e-5, 3e-4)
    return dict(output=compare(records[1][0], records[0][0], atol, rtol),
                gradients=compare(records[1][1], records[0][1], atol, rtol),
                gradients_by_tensor={name: compare([a], [b], atol, rtol)
                                     for name, a, b in zip(names, records[1][1], records[0][1])},
                audit=audit(model, h, pos, edges, alpha, geom))


def timed(model, h, pos, edges, alpha, geom, train, pathwise):
    model.pathwise = pathwise
    model.zero_grad(set_to_none=True)
    for v in (*h.values(), alpha, *geom.values()):
        v.grad = None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.set_grad_enabled(train):
        y = model(h, pos, edges, alpha, centered_positions=pos, node_geometry=geom)
        if train:
            sum(v.square().mean() for v in y.values()).backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter()-start)*1000
    peak = torch.cuda.max_memory_allocated()
    return dict(ms=elapsed, base_mib=base/2**20, peak_mib=peak/2**20,
                increment_mib=(peak-base)/2**20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT / 'config.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert cfg['bandwidth'] is None and cfg['outer_path_cap'] is None
    assert cfg['aggregation_backend'] == 'sparse' and cfg['precision'] == 'float32'
    assert cfg['tf32'] is False and cfg['geometry_precomputed'] is True
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'preserve previous results: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = [Path(__file__), ROOT/'shr_operator/shr.py', ROOT/'shr_operator/paths.py',
               ROOT/'shr_operator/primitives.py', ROOT/'shr_operator/aggregation.py', Path(args.config)]
    hashes = {str(f.resolve().relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    report = dict(experiment_id='SHR-PAPER-01-A', config=cfg,
                  source_hashes=hashes, torch=torch.__version__,
                  cuda=torch.version.cuda, e3nn=e3nn.__version__,
                  gc_policy='collect after validation; disable cyclic GC inside timing; refcount freeing remains enabled',
                  rows=[], validation=[], failures=[])
    output.write_text(json.dumps(report, indent=2))
    for G in cfg['G']:
        try:
            torch.manual_seed(cfg['seed'])
            model = AblationSHR(feature_lmax=cfg['L'], in_channels=cfg['channels'],
                                geometry_orders=range(G+1), B=None, K=None, aggregation_backend='sparse').double()
            h, pos, edges, alpha = inputs(cfg, torch.double, 'cpu', n=12)
            check = dict(G=G, cpu_fp64=validate(model, h, pos, edges, alpha, G), manifest=model.plan.manifest())
            report['validation'].append(check)
            print(f'G={G} FP64 validated', flush=True)
            if not args.validate_only:
                model = model.float().cuda()
                h, pos, edges, alpha = inputs(cfg, torch.float32, 'cuda')
                check['cuda_fp32'] = validate(model, h, pos, edges, alpha, G)
                gc.collect()
                geom = {l: g.detach().requires_grad_() for l, g in regular_solid_harmonics(pos, G).items()}
                for train in (False, True):
                    samples = {False: [], True: []}
                    gc.collect()
                    gc.disable()
                    try:
                        for r in range(cfg['warmup']+cfg['repeats']):
                            for pathwise in ((False, True) if r%2 == 0 else (True, False)):
                                sample = timed(model, h, pos, edges, alpha, geom, train, pathwise)
                                if r >= cfg['warmup']:
                                    samples[pathwise].append(sample)
                    finally:
                        gc.enable()
                    for pathwise, values in samples.items():
                        times = [v['ms'] for v in values]
                        row = dict(G=G, mode='forward_backward' if train else 'forward',
                                   variant='pathwise' if pathwise else 'shared', mean_ms=statistics.mean(times),
                                   std_ms=statistics.stdev(times), median_ms=statistics.median(times),
                                   peak_mib=max(v['peak_mib'] for v in values),
                                   increment_mib=max(v['increment_mib'] for v in values), samples=values)
                        report['rows'].append(row)
                        print(json.dumps({k: v for k, v in row.items() if k != 'samples'}), flush=True)
                model.zero_grad(set_to_none=True)
                del geom
            del model, h, pos, edges, alpha
        except Exception as exc:
            report['failures'].append(dict(G=G, error=repr(exc)))
            output.write_text(json.dumps(report, indent=2))
            raise
        output.write_text(json.dumps(report, indent=2))
    report['source_hashes_unchanged'] = all(hashlib.sha256(f.read_bytes()).hexdigest() == hashes[str(f.resolve().relative_to(ROOT))] for f in sources)
    assert report['source_hashes_unchanged'], 'source changed during benchmark'
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
