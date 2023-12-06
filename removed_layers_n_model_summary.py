# """ Model look good but error produce while training
from transformers import WhisperConfig, WhisperModel
from transformers import WhisperForConditionalGeneration
import torch
from torchsummary import summary

model_name = "openai/whisper-large-v2"
model = WhisperForConditionalGeneration.from_pretrained(model_name)



################


# Set the input shape for the audio task
# batch_size = 1
# sequence_length = 1500  # Adjust this based on your requirements
# num_mel_features = 80

# # Create a dummy input tensor
# dummy_input = torch.randn(batch_size, num_mel_features, sequence_length)

# # Print the model summary
# summary(model, dummy_input)


def model_summary(model):
    print("Model Summary:")
    print("--------------------")
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        # print("{:30} {:>12} {:>15}".format(name, str(tuple(param.shape)), param.numel()))

    print("--------------------")
    print("Total Parameters:    ", total_params)
    print("Trainable Parameters:", trainable_params)

# Instantiate the model


# Print the model summary
model_summary(model)


################
# print(model)



# summary(model, (1280,80,3))


config = model.config


desired_vocab_size = config.vocab_size  #config.vocab_size #3397
# config.vocab_size = desired_vocab_size
# config.pad_token_id = desired_vocab_size - 1

new_model = WhisperForConditionalGeneration(config)

pretrained_model_dict = model.state_dict()
new_model_dict = new_model.state_dict()

pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if "decoder.embed_tokens" not in k}

new_proj_out_weight = torch.randn((desired_vocab_size, config.hidden_size))
new_proj_out_bias = torch.zeros((desired_vocab_size,))
new_model_dict["decoder.proj.weight"] = torch.nn.parameter.Parameter(new_proj_out_weight)
new_model_dict["decoder.proj.bias"] = torch.nn.parameter.Parameter(new_proj_out_bias)

pretrained_proj_out_weight = pretrained_model_dict["proj_out.weight"]

# Remove the mismatched proj_out.weight from pretrained_model_dict
del pretrained_model_dict["proj_out.weight"]

new_model_dict.update(pretrained_model_dict)

new_model.load_state_dict(new_model_dict, strict=False)

# Copy the pre-trained values into the new projection layer weight only for the dimensions that match the desired vocabulary size
with torch.no_grad():
    new_model.proj_out.weight[:pretrained_proj_out_weight.size(0), :].copy_(pretrained_proj_out_weight[:desired_vocab_size, :])

model = new_model



# print(model.config)






# # """
# ## 15. Override generation arguments
# model.config.apply_spec_augment = apply_spec_augment
# model.config.dropout = dropout
# model.config.forced_decoder_ids = None
# model.config.suppress_tokens = []
# if gradient_checkpointing:
#     model.config.use_cache = False
# if freeze_feature_encoder:
#     model.freeze_feature_encoder()






## 16 Truncate_model
import copy

def truncate_model(model, d_layers_to_remove=0, e_layers_to_remove=0):
  print(f"e_layers_to_remove {e_layers_to_remove}, d_layers_to_remove {d_layers_to_remove}")
  num_e_layers = len(model.model.encoder.layers)
  num_d_layers = len(model.model.decoder.layers)

#   print(num_e_layers, num_d_layers)
  model_truncated = copy.deepcopy(model)

#   print(model.model.encoder.layers.children())

  model_truncated.model.encoder.layers = torch.nn.ModuleList(list(model.model.encoder.layers.children()))[:num_e_layers-e_layers_to_remove]

#   print(model_truncated.model.encoder.layers)
  model_truncated.model.decoder.layers = torch.nn.ModuleList(list(model.model.decoder.layers.children()))[:num_d_layers-d_layers_to_remove]
  return model_truncated

# def truncate_model(model, d_layers_to_remove=0):
#   num_d_layers = len(model.model.decoder.layers)
#   model_truncated = copy.deepcopy(model)
#   model_truncated.model.decoder.layers = torch.nn.ModuleList(list(model.model.decoder.layers.children()))[:num_d_layers-d_layers_to_remove]
#   return model_truncated

truncated_model = truncate_model(model, d_layers_to_remove = 12, e_layers_to_remove=12)
model = truncated_model

# print(model)

model_summary(model)


model_name = "openai/whisper-medium"
model = WhisperForConditionalGeneration.from_pretrained(model_name)

model_summary(model)
