from gym import spaces
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet import VisionNetwork

from ray.rllib.utils.framework import try_import_tf

tf = try_import_tf()

import pdb


class ActionMaskModel(TFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        super(ActionMaskModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        # vision network
        actual_obs_space = model_config["custom_model_config"]["actual_obs_space"]
        self.base_model = VisionNetwork(actual_obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.base_model.variables())


    def forward(self, input_dict, state, seq_lens):
        actual_obs_dict = input_dict.copy()
        actual_obs_dict["obs"] = input_dict["obs"]["actual_obs"]

        model_out, state = self.base_model(actual_obs_dict, state, seq_lens)
        print(model_out)
        inf_mask = tf.maximum(tf.log(input_dict["obs"]["action_mask"]), tf.float32.min)
        # also mask out eviction location choicce if agent not selected
        return model_out + inf_mask, state


    def value_function(self):
        return self.base_model.value_function()
