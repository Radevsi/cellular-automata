#@title Initialize Training { vertical-output: true}

print('\n...........................IN main.py...........................')

# https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import tensorflow as tf
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import time
import zipfile

from image_processing import load_local_image, load_alive_image
from model import CAModel, load_emoji, to_rgb
from utils import imshow, zoom, export_ca_to_webgl_demo, manage_dir
from train import train_ca
from figures import FigGen
from experiments import Experiments

print('...........................FINISHED IMPORTS...........................')

def main():

    # Sort out which gpu to use
    with tf.device('/CPU:0'):
        physical_devices = tf.config.list_physical_devices('GPU') 
        gpus = (physical_devices != [])
        n_steps = 8000 # training steps
    if gpus:
        print("Num Physical GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    else:
        print("WARNING: Running without GPUs")
        n_steps = 1

    # Cellular Automata Parameters
    HIDDEN_SIZE = 512 # size of hidden layer in CNN
    CHANNEL_N = 40 # number of CA state channels
    TARGET_PADDING = 16 # number of pixels used to pad the target image border
    BATCH_SIZE = 8
    POOL_SIZE = 1024
    CELL_FIRE_RATE = 0.5
    STEP_SIZE = 1.0
    THRESHOLD = 0.01 # hyper-parameter denoting threshold for life

    EXPERIMENT_TYPE = "persistent" #@param ["Growing", "Persistent", "Regenerating"]
    EXPERIMENT_MAP = {"growing":0, "persistent":1, "regenerating":2}
    EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

    USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
    DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # number of patterns to damage in a batch

    # Booleans to decide what information to store
    MAKE_CHECKPOINTS = False
    MAKE_POOL = False

    # Run experiments parameter
    EXPERIMENT_NUMBER = 3
    RUN_EXPERIMENTS = True

    if RUN_EXPERIMENTS:
 
        LIVING_MAP = {"bob-ross-painting":1, "starry-night":1, 
                      "mozart1":0, "sleigh":0, "ca-pyramid":0,
                      "mozart":1}

        # Run experiments from experiments module
        experiments = Experiments(EXPERIMENT_TYPE, CELL_FIRE_RATE, STEP_SIZE, 
                HIDDEN_SIZE, CHANNEL_N, TARGET_PADDING, BATCH_SIZE, POOL_SIZE, 
                USE_PATTERN_POOL, DAMAGE_N, THRESHOLD, LIVING_MAP, n_steps, MAKE_POOL)

        ## Experiment 1
        if EXPERIMENT_NUMBER == 1:
            # image_names = ['bob-ross-painting', 'ca-pyramid', 'ca-pyramid', 'ca-pyramid']
            # target_sizes = [135, 300, 300, 300]
            # model_params = [(12, 100), (28, 300), (24, 160), (28, [64, 128, 32])]

    #         image_names = ['starry-night', 'starry-night', 'starry-night', 'starry-night']
    #         target_sizes = [150, 150, 150, 150]
    #         model_params = [(24, 128), (12, 100), (32, 100), (24, [256, 300])]

            image_names = ['starry-night']
            target_sizes = [150]
            model_params = [(12, 100)]

            # image_names = ['mozart', 'mozart1', 'mozart', 'mozart1', 'mozart', 'mozart1', 'mozart', 'mozart1']
            # target_sizes = [48, 48, 48, 48, 48, 48, 48, 48]
            # model_params = [(20, 128), (20, 128), (20, 160), (20, 160), (24, 184), (24, 184), (24, 256), (24, 256)]

            # image_names = ['starry-night', 'starry-night', 'starry-night', 'bob-ross-painting', 'starry-night', 'sleigh']
            # target_sizes = [150, 150, 150, 135, 150, 48]
            # model_params = [(40, 300), (24, 128), (32, 100), (24, [256, 300]), (12, 100), (24, 256)]
            experiments.experiment1(image_names=image_names, target_sizes=target_sizes, model_params=model_params)

            
        ## Experiment 3 (toggling the cell fire rate)
        if EXPERIMENT_NUMBER == 3:

            image_name = 'bob-ross-painting'
            target_size = 135
            channel_n, hidden_size = 24, 256
            cell_fire_rates = [0.3, 0.7, 0.5, 0.4]
            
            experiments.experiment3(image_name=image_name, target_size=target_size, channel_n=channel_n, 
                                    hidden_size=hidden_size, cell_fire_rates=cell_fire_rates)

        ## Experiment 4 (toggling the step size)
        if EXPERIMENT_NUMBER == 4:

            image_name = 'bob-ross-painting'
            target_size = 135
            channel_n, hidden_size = 12, 100
            step_sizes = [0.3, 1.0, 2.0, 3.0]
            
            experiments.experiment4(image_name=image_name, target_size=target_size, channel_n=channel_n, 
                                    hidden_size=hidden_size, step_sizes=step_sizes)                                    

        return 0

    # Select image to run on
    image_name = 'bob-ross-painting'
    # TARGET_EMOJI = '🛩'
    TARGET_EMOJI = None
    TARGET_SIZE = 125
    
    output_dir = f'figures/{image_name}-{TARGET_SIZE}/{EXPERIMENT_TYPE}/channel-{CHANNEL_N}_hidden-{HIDDEN_SIZE}'

    if TARGET_EMOJI != None:
        target_img = load_emoji(TARGET_EMOJI)
        print(f'target emoji is {TARGET_EMOJI}')
        output_dir = f'figures/{TARGET_EMOJI}-{TARGET_SIZE}/{EXPERIMENT_TYPE}/channel-{CHANNEL_N}_hidden-{HIDDEN_SIZE}'
        imshow(zoom(to_rgb(target_img), 2), fmt='png', SAVE=True, path=output_dir)
        # print(f'target image is: {target_img}')
    else: 
        manage_dir(output_dir=output_dir+'/train_log', remove_flag=True) 
        target_img, _alpha_channel, _orig_img = load_alive_image(image_name, max_size=TARGET_SIZE)
        print(f"Using image {image_name}.png with max_size of {TARGET_SIZE}")

    # Run the regular model
    # Initialize model
    ca = CAModel(channel_n=CHANNEL_N, hidden_size=HIDDEN_SIZE, fire_rate=CELL_FIRE_RATE)
    ca.dmodel.summary()

    # Train it
    start_time = time.time()
    _ = train_ca(ca, image_name, target_size, target_img, CHANNEL_N, HIDDEN_SIZE, TARGET_PADDING, BATCH_SIZE,
            POOL_SIZE, USE_PATTERN_POOL, DAMAGE_N, steps=n_steps, path=output_dir,
            make_pool=MAKE_POOL)
    print(f"\nTraining took {time.time() - start_time} seconds")

    # Save some figures of training progress
    fig_gen = FigGen(ca, output_dir)
    steps = [100, 500, 800, 1000] if gpus else [0]
    fig_gen.training_progress_batches()

    if MAKE_CHECKPOINTS:
        fig_gen.training_progress_checkpoints(damage_n=DAMAGE_N, channel_n=CHANNEL_N, steps=steps)

    if MAKE_POOL:
        fig_gen.pool_contents()

    # # Export quantized model for WebGL demo
    # if gpus:
    #     model = CAModel(channel_n=CHANNEL_N, hidden_size=HIDDEN_SIZE, fire_rate=CELL_FIRE_RATE)
    #     model.load_weights(output_dir+'/train_log/%04d'%n_steps)
    #     with zipfile.ZipFile(output_dir+'_webgl.zip', 'w') as zf:
    #         zf.writestr(f'channel-{CHANNEL_N}_hidden-{HIDDEN_SIZE}.json', export_ca_to_webgl_demo(model))

if __name__ == '__main__':
    main()
