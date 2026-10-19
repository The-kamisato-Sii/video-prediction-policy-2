import torch
from vpp2.libero.diagnostics import ActionDiagnosticEvaluator


def test_short_decode_keeps_logical_inference_horizon():
    sample = dict(video=torch.zeros(3, 1, 16, 32), video_num_frames=17,
                  action=torch.zeros(32, 7), proprio=torch.zeros(1, 8), prompt='test')

    class Model:
        device = 'cpu'

        def training_loss(self, batch):
            assert batch['video_num_frames'] == 17
            return torch.tensor(0.), {}

        def infer_action(self, **kwargs):
            assert kwargs['num_video_frames'] == 17
            assert kwargs['action_horizon'] == 32
            return {'action': torch.zeros(32, 7)}

    result = ActionDiagnosticEvaluator([sample], [0]).evaluate(Model(), 200)
    assert result == {'action/val_loss': 0., 'action/l1': 0.}
