"""CPU contract checks, including Accelerate's real distributed batch sharder."""
import unittest
from collections import Counter
import torch
from torch.utils.data import Dataset, Subset, BatchSampler
from accelerate.data_loader import BatchSamplerShard
from vpp2.libero_flow.sampler import HomogeneousTaskSampler, task_indices

class MetadataDataset(Dataset):
    def __init__(self, counts=(17, 87, 14, 23)):
        self.episodes=[];self.ends=[];n=0
        for i,count in enumerate(counts):
            n+=count;self.episodes.append({'indicator':i%2});self.ends.append(n)
    def __len__(self):return self.ends[-1]
    def __getitem__(self,index):raise AssertionError('Sampler must not read payloads')

class TestSampler(unittest.TestCase):
    def test_coverage_sharding_resume(self):
        ds=MetadataDataset()
        labels={i:t for t,p in enumerate(task_indices(ds)) for i in p.tolist()}
        for world,gas in [(1,1),(2,1),(2,3),(16,1)]:
            sampler=HomogeneousTaskSampler(ds,42,2,world,gas)
            ids=list(sampler)
            self.assertEqual(len(ids),len(sampler));self.assertEqual(set(ids),set(range(len(ds))))
            counts=Counter(labels[i] for i in ids)
            for task in (0,1):
                self.assertEqual(counts[task],sampler.blocks[task]*sampler.block_size)
                self.assertLess(sampler.audit['repeated_tail_samples'][task],sampler.block_size)
            shards=[list(BatchSamplerShard(BatchSampler(sampler,2,False),num_processes=world,process_index=r,split_batches=False,even_batches=True)) for r in range(world)]
            for k in range(len(shards[0])):
                self.assertEqual({labels[i] for shard in shards for i in shard[k]}, {labels[shards[0][k][0]]})
            for k in range(0,len(shards[0]),gas):
                self.assertEqual(len({labels[i] for batch in shards[0][k:k+gas] for i in batch}),1)
            sampler.set_epoch(3);epoch3=list(sampler)
            restored=HomogeneousTaskSampler(ds,42,2,world,gas)
            restored.set_epoch_offset(3);restored.set_resume_batch_offset(gas)
            self.assertEqual(list(restored),epoch3[gas*2*world:])
            restored.clear_resume_batch_offset();restored.set_epoch(1)
            sampler.set_epoch(4);self.assertEqual(list(restored),list(sampler))
            self.assertNotEqual(epoch3,ids)
    def test_subset_and_single_task(self):
        ds=MetadataDataset();sub=Subset(Subset(ds,[0,20,35,100,140]),[4,1,0,1])
        self.assertEqual([x.tolist() for x in task_indices(sub)],[[2],[0,1,3]])
        for task in (0,1):
            sub=Subset(ds,task_indices(ds)[task].tolist())
            sampler=HomogeneousTaskSampler(sub,2,16,2)
            self.assertEqual(set(sampler),set(range(len(sub))))

if __name__=='__main__':unittest.main()
