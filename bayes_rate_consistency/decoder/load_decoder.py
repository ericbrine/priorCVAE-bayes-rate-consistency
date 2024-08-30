import os
from flax import linen as nn

# from bayes_rate_consistency.decoder.decoder import MLPDecoder
from priorCVAE.models import MLPDecoder
from priorCVAE.utility import load_model_params


def load_decoder(project_root, model_args):
    decoder_path = os.path.join(project_root, model_args['decoder_path'])
    decoder_params = load_model_params(decoder_path)["decoder"]
    decoder = MLPDecoder([model_args['hidden_dim3'], model_args['hidden_dim2'], model_args['hidden_dim1']], model_args['input_dim'], activations=nn.gelu)

    return decoder, decoder_params