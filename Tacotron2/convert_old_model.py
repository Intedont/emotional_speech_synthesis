import torch

model_dict = torch.load('checkpoints/tacotron2_1032590_6000_amp')
new_state_dict = {}
for k, v in model_dict['state_dict'].items():
    print(k, v)
    new_state_dict[k[7:]] = v
model_dict['state_dict'] = new_state_dict
torch.save(model_dict, 'new_model')