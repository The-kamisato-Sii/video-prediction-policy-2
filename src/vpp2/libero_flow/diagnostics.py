"""Fixed-seed10-step action and decoded-motion diagnostics."""
import json
from pathlib import Path
import torch
from .data import collate_flow_action


class FlowActionDiagnosticEvaluator:
    def __init__(self, dataset, output, seed=42):
        self.dataset, self.output, self.seed = dataset, Path(output), int(seed)
        start=dataset.first_index(0)
        self.action_indices=list(range(start,start+4))
        self.flow_index=None if dataset.audit["flow_starts"] == 0 else dataset.first_index(1)

    @torch.no_grad()
    def evaluate(self, model, step):
        device=torch.device(model.device)
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(self.seed)
            errors=[]
            for index in self.action_indices:
                batch=collate_flow_action([self.dataset[index]])
                output=model.sample_task(batch,indicator=0,steps=10)["action"].float()
                target=batch["action"].to(output)
                valid=(~batch["action_is_pad"]).to(output.device)[:,:,None]
                errors.append(float(model._masked_mean((output-target).square(),valid).mean()))
            result={"action_mse":sum(errors)/len(errors)}
            if self.flow_index is not None:
                batch=collate_flow_action([self.dataset[self.flow_index]])
                output=model.sample_task(batch,indicator=1,steps=getattr(model,"flow_inference_steps",10))
                for channel,key in enumerate(("point_map","scene_flow")):
                    prediction=output[key].float()
                    target=batch["motion"][:,:,channel*3:(channel+1)*3].to(prediction)
                    valid=batch["motion_valid"][:,:,channel:channel+1].to(prediction.device)
                    result[f"{key}_rmse"]=float(model._masked_mean((prediction-target).square(),valid).mean().sqrt())
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/f"step_{step:06d}.json").write_text(json.dumps(result,indent=2)+'\n')
        return result
