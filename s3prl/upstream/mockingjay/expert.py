# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


from collections import OrderedDict
from typing import List, Tuple

import torch
import yaml
from torch import Tensor

from ..interfaces import UpstreamBase
from .builder import PretrainedTransformer


class UpstreamExpert(UpstreamBase):
    """
    The Mockingjay wrapper
    """

    def __init__(self, ckpt, options_config=None, **kwargs):
        super().__init__(**kwargs)

        if options_config is not None:
            print(
                "[UpstreamExpert] - Using upstream expert config file from:",
                options_config,
            )
            with open(options_config, "r") as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print("[UpstreamExpert] - Using the default upstream expert config")
            options = {
                "load_pretrain": "True",
                "no_grad": "False",
                "dropout": "default",
                "spec_aug": "False",
                "spec_aug_prev": "True",
                "output_hidden_states": "True",
                "permute_input": "False",
            }

        options["ckpt_file"] = ckpt
        options["select_layer"] = -1

        self.transformer = PretrainedTransformer(options, inp_dim=-1)
        assert hasattr(
            self.transformer, "extracter"
        ), "This wrapper only supports `on-the-fly` ckpt with built in feature extracters."
        self.transformer([torch.randn(16000)])

    def get_downsample_rates(self, key: str) -> int:
        return 160

    # def forward(self, wavs):
    #     last_hidden_state, hidden_states = self.transformer(
    #         wavs
    #     )  # (batch_size, extracted_seqlen, feature_dim)

    #     print("---------")
    #     print("Len supposed last hidden state: ", len(last_hidden_state))
    #     print("type of last hidden state: ", type(last_hidden_state))

    #     print("---------")
    #     print("Len hidden_states: ", len(hidden_states))
    #     print("Type of hidden_states: ", type(hidden_states))
    #     print("first item of hidden states: ", hidden_states[0])
    #     print("len after unbinding hidden states: ", len(hidden_states.unbind(dim=0)))
    #     print("Type of after unbinding hidden_states: ", type(hidden_states.unbind(dim=0)))
    #     print("first item after unbinding of hidden states: ", hidden_states.unbind(dim=0)[0])




    #     return {
    #         "last_hidden_state": last_hidden_state,
    #         "hidden_states": hidden_states.unbind(dim=0),
    #     }


    # updated for attentions
    def forward(self, wavs):
        last_hidden_layer, encoded_layers = self.transformer(
            wavs
        )  # (batch_size, extracted_seqlen, feature_dim)

        attentions = encoded_layers[0]
        keys = encoded_layers[1]
        queries = encoded_layers[2]
        hidden_states = torch.stack(encoded_layers[1])

        print("---------")
        print("Len of attentions: ", len(attentions))
        print("Type of attentions: ", type(attentions))

        print("---------")
        print("Len encoded_layers: ", len(encoded_layers))
        print("Type encoded_layers: ", type(encoded_layers))
        print("Len encoded_layers[0] (same as attentions): ", len(encoded_layers[0]))
        print("Type of encoded_layers[0] (same as attentions): ", type(encoded_layers[0]))
        print("Len of encoded_layers[1] : ", type(encoded_layers[1]))
        print("Len of encoded_layers[2] : ", type(encoded_layers[2]))

        # this type should be a tensor! where is the type being changed?
        print("Len encoded_layers[1] (hidden_states): ", len(encoded_layers[1]))
        print("Type of encoded_layers[1] (hidden_states): ", type(encoded_layers[1]))

        # print("inside EXPERT: last item in encoded layers [3]: ")
        # print(hidden_states[3][0])
        
        return {
            "last_hidden_layer": last_hidden_layer,
            "attention_encoding": attentions, 
            "keys": keys,
            "queries": queries,
            "hidden_states": hidden_states.unbind(dim=0)
        }
