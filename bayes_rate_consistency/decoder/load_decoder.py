import os

from bayes_rate_consistency.decoder.decoder import MLPDecoder
from priorCVAE.utility import load_model_params


def load_decoder(project_root, decoder_path, hidden_dim, input_dim):

    decoder_path = os.path.join(project_root, decoder_path)
    decoder_params = load_model_params(decoder_path)["decoder"]
    decoder = MLPDecoder(hidden_dim, input_dim)

    return decoder, decoder_params