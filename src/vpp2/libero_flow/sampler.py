"""Shuffle task samples, then interleave homogeneous global optimizer batches."""
import math

import torch
from torch.utils.data import Subset

from vpp2.utils.samplers import ResumableEpochSampler


def task_indices(dataset):
    """Read metadata only; never load video, motion or action labels."""
    if isinstance(dataset, Subset):
        parent = task_indices(dataset.dataset)
        labels = torch.empty(len(dataset.dataset), dtype=torch.uint8)
        for task, ids in enumerate(parent):
            labels[ids] = task
        selected = labels[torch.as_tensor(dataset.indices, dtype=torch.long)]
        return [torch.where(selected == task)[0] for task in (0, 1)]
    pools = [[], []]
    start = 0
    for record, end in zip(dataset.episodes, dataset.ends):
        pools[int(record['indicator'])].append(torch.arange(start, end))
        start = end
    if start != len(dataset):
        raise ValueError('Task metadata does not cover dataset')
    return [torch.cat(parts) if parts else torch.empty(0, dtype=torch.long) for parts in pools]


class HomogeneousTaskSampler(ResumableEpochSampler):
    """Accelerate shards consecutive local batches across ranks.

    Every global optimizer batch (including accumulation) has one task. Each
    task is shuffled independently and exhausted once per epoch. Its last block
    repeats shuffled samples to keep every rank/microbatch full; no samples drop.
    Resume offsets have the same per-rank microbatch meaning as the baseline.
    """
    def __init__(self, dataset, seed, batch_size, num_processes, accumulation_steps=1):
        super().__init__(dataset, seed, batch_size, num_processes)
        self.block_size = int(batch_size) * int(num_processes) * int(accumulation_steps)
        if self.block_size < 1:
            raise ValueError('Global optimizer batch must be positive')
        self.pools = task_indices(dataset)
        self.blocks = [math.ceil(len(p) / self.block_size) for p in self.pools]
        self.audit = dict(policy='homogeneous_interleaved', global_block_size=self.block_size,
                          task_samples=[len(p) for p in self.pools], task_blocks=self.blocks,
                          repeated_tail_samples=[n*self.block_size-len(p) for n,p in zip(self.blocks,self.pools)],
                          task_order=['action', 'flow'])

    def __len__(self):
        return sum(self.blocks) * self.block_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch + self.epoch_offset)
        shuffled = []
        for pool, count in zip(self.pools, self.blocks):
            ids = pool[torch.randperm(len(pool), generator=g)]
            required = count * self.block_size
            if required:
                ids = ids.repeat(math.ceil(required / len(ids)))[:required]
            shuffled.append(ids)
        # Evenly spread each task's blocks over the epoch, then rotate the
        # schedule randomly. Within-task sample permutations change every epoch.
        schedule = sorted(((j + 0.5) / n, task, j)
                          for task, n in enumerate(self.blocks) for j in range(n))
        if schedule:
            shift = int(torch.randint(len(schedule), (1,), generator=g))
            schedule = schedule[shift:] + schedule[:shift]
        skip = self.resume_batch_offset*self.batch_size*self.num_processes if self.epoch == 0 else 0
        for _, task, j in schedule:
            block = shuffled[task][j*self.block_size:(j+1)*self.block_size]
            if skip >= len(block):
                skip -= len(block)
                continue
            yield from block[skip:].tolist()
            skip = 0
