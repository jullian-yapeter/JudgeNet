from __future__ import print_function

# import tensorflow.compat.v1 as tf
import tensorflow.compat.v1 as tf

import json
from bs4 import BeautifulSoup
import requests

import data_processing.vggish.vggish_input as vggish_input
import data_processing.vggish.vggish_params as vggish_params
import data_processing.vggish.vggish_postprocess as vggish_postprocess
import data_processing.vggish.vggish_slim as vggish_slim

vggish_root = "data_processing/vggish"

def extractVGGish(fileName):
  examples_batch = vggish_input.wavfile_to_examples(fileName)

  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(f'{vggish_root}/vggish_pca_params.npz')

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, f'{vggish_root}/vggish_model.ckpt')
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    postprocessed_batch = pproc.postprocess(embedding_batch)
  return postprocessed_batch

def scrapeVideoURL(url):
    # Get video url from raw html
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.find(id="__NEXT_DATA__")
    parsed = json.loads(text.string)

    playerData = parsed["props"]["pageProps"]["videoData"]["playerData"]
    parsed2 = json.loads(playerData)

    video_url = parsed2["resources"]["h264"][0]["file"]
    return video_url
