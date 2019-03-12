import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
    name='vocoder',
    cleaners='english_cleaners',
    builder="wavenet",
    tacotron_gpu_start_idx=0,
    tacotron_num_gpus=1,
    wavenet_gpu_start_idx=0,
    wavenet_num_gpus=1,
    split_on_cpu=True,
    num_mels=80,
    num_freq=1025,
    rescaling=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.999,  # Rescaling value
    trim_silence=True,
    clip_mels_length=True,
    max_mel_frames=1000,
    use_lws=False,
    silence_threshold=2,  # silence threshold used for sound trimming
    fft_size=1024,
    hop_size=256,
    sample_rate=22050,
    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
    trim_fft_size=512,
    trim_hop_size=128,
    trim_top_db=23,
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=False,
    max_abs_value=1.,
    normalize_for_wavenet=True,
    clip_for_wavenet=True,
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.
    min_level_db=-100,
    ref_level_db=20,
    fmin=125,
    fmax=7600,  # To be increased/reduced depending on data.
    power=1.5,
    griffin_lim_iters=60,
    outputs_per_step=1,
    stop_at_any=True,
    embedding_dim=512,  # dimension of embedding space
    enc_conv_num_layers=3,  # number of encoder convolutional layers
    enc_conv_kernel_size=(5, ),
    enc_conv_channels=512,  # number of encoder convolutions filters
    encoder_lstm_units=256,
    smoothing=False,  # Whether to smooth the attention normalization function
    attention_dim=128,  # dimension of attention space
    attention_filters=32,  # number of attention convolution filters
    attention_kernel=(31, ),  # kernel size of attention convolution
    cumulative_weights=True,
    prenet_layers=[256, 256],  # number of layers and number of units of prenet
    decoder_layers=2,  # number of decoder lstm layers
    decoder_lstm_units=1024,  # number of decoder lstm units on each layer
    max_iters=2000,
    postnet_num_layers=5,  # number of postnet convolutional layers
    postnet_kernel_size=(5, ),
    postnet_channels=512,  # number of postnet convolution filters for each
    cbhg_kernels=8,
    cbhg_conv_channels=128,  # Channels of the convolution bank
    cbhg_pool_size=2,  # pooling size of the CBHG
    cbhg_projection=256,
    cbhg_projection_kernel_size=3,  # kernel_size of the CBHG projections
    cbhg_highwaynet_layers=4,  # Number of HighwayNet layers
    cbhg_highway_units=128,
    cbhg_rnn_units=128,
    mask_encoder=True,
    mask_decoder=False,
    cross_entropy_pos_weight=20,
    predict_linear=True,
    input_type="raw",
    quantize_channels=2 ** 16,
    log_scale_min=float(np.log(1e-14)),
    log_scale_min_gauss=float(np.log(1e-7)),
    out_channels=2,
    layers=20,
    stacks=2,
    residual_channels=128,  # Number of residual block input/output channels.
    gate_channels=256,  # split in 2 in gated convolutions
    skip_out_channels=128,
    kernel_size=3,  # The number of inputs to consider in dilated convolutions.
    cin_channels=80,
    upsample_conditional_features=True,
    upsample_type='1D',
    upsample_activation='LeakyRelu',
    upsample_scales=[5, 5, 11],
    freq_axis_kernel_size=3,
    leaky_alpha=0.4,
    gin_channels=-1,
    use_speaker_embedding=True,  # whether to make a speaker embedding
    n_speakers=5,  # number of speakers (rows of the embedding)
    use_bias=True,
    max_time_sec=None,
    max_time_steps=11000,
    tacotron_random_seed=5339,
    tacotron_data_random_state=1234,
    tacotron_swap_with_cpu=False,
    tacotron_batch_size=32,
    tacotron_synthesis_batch_size=1,
    tacotron_test_size=0.05,
    tacotron_test_batches=None,  # number of test batches.
    tacotron_decay_learning_rate=True,
    tacotron_start_decay=50000,  # Step at which learning decay starts
    tacotron_decay_steps=50000,
    tacotron_decay_rate=0.5,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-5,  # minimal learning rate
    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer Epsilon parameter
    tacotron_reg_weight=1e-7,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=False,
    tacotron_zoneout_rate=0.1,
    tacotron_dropout_rate=0.5,
    tacotron_clip_gradients=True,  # whether to clip gradients
    natural_eval=False,
    tacotron_teacher_forcing_mode='constant',
    tacotron_teacher_forcing_ratio=1.,
    tacotron_teacher_forcing_init_ratio=1.,
    tacotron_teacher_forcing_final_ratio=0.,
    tacotron_teacher_forcing_start_decay=10000,
    tacotron_teacher_forcing_decay_steps=280000,
    tacotron_teacher_forcing_decay_alpha=0.,
    wavenet_random_seed=5339,  # S=5, E=3, D=9 :)
    wavenet_data_random_state=1234,
    wavenet_swap_with_cpu=False,
    wavenet_batch_size=8,  # batch size used to train wavenet.
    wavenet_synthesis_batch_size=10 * 2,
    wavenet_test_size=0.0441,
    wavenet_test_batches=None,  # number of test batches.
    wavenet_lr_schedule='exponential',
    wavenet_learning_rate=1e-4,  # wavenet initial learning rate
    wavenet_warmup=float(4000),
    wavenet_decay_rate=0.5,
    wavenet_decay_steps=300000,
    wavenet_adam_beta1=0.9,  # Adam beta1
    wavenet_adam_beta2=0.999,  # Adam beta2
    wavenet_adam_epsilon=1e-8,  # Adam Epsilon
    wavenet_clip_gradients=False,
    wavenet_ema_decay=0.9999,  # decay rate of exponential moving average
    wavenet_weight_normalization=False,
    wavenet_init_scale=1.,
    wavenet_dropout=0.05,  # drop rate of wavenet layers
    train_with_GTA=False,
    dropout=1 - 0.95,
    weight_normalization=True,
    legacy=True,
    pin_memory=True,
    num_workers=2,
    test_size=0.0441,  # 50 for CMU ARCTIC single speaker
    test_num_samples=10,
    random_state=1234,
    batch_size=2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    amsgrad=False,
    initial_learning_rate=1e-3,
    lr_schedule="noam_learning_rate_decay",
    lr_schedule_kwargs={},  # {"anneal_rate": 0.5, "anneal_interval": 50000},
    nepochs=2000,
    weight_decay=0.0,
    clip_thresh=-1,
    exponential_moving_average=True,
    ema_decay=0.9999,
    checkpoint_interval=10000,
    train_eval_interval=10000,
    test_eval_epoch_interval=5,
    save_optimizer_state=True,
    sentences=[
        # From July 8, 2017 New York Times:
        'Scientists at the CERN laboratory say they have discovered a new\
        particle.',
        'There\'s a way to measure the acute emotional intelligence that has\
                never gone out of style.',
        'President Trump met with other leaders at\
                the Group of 20 conference.',
        'The Senate\'s bill to repeal and replace the Affordable\
                Care Act is now imperiled.',
        # From Google's Tacotron example page:
        'Generative adversarial network or variational auto-encoder.',
        'Basilar membrane and otolaryngology are not auto-correlations.',
        'He has read the whole thing.',
        'He reads books.',
        'He thought it was time to present the present.',
        'Thisss isrealy awhsome.',
        'Punctuation sensitivity, is working.',
        'Punctuation sensitivity is working.',
        "Peter Piper picked a peck of pickled peppers.\
                How many pickled peppers did Peter Piper pick?",
        "She sells sea-shells on the sea-shore.\
                The shells she sells are sea-shells I'm sure.",
        "Tajima Airport serves Toyooka.",
        # From The web (random long utterance)
        'Sequence to sequence models have enjoyed great success\
                in a variety of tasks such as machine\
                translation, speech recognition, and text summarization.\
                This project covers a sequence to\
                sequence model trained to predict a\
                speech representation from an input sequence of\
                characters. We show that\
                the adopted architecture is\
                able to perform this task with wild success.',
        'Thank you so much for your support!',
    ]

)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name])
          for name in sorted(values) if name != 'sentences']
    return 'Hyperparameters:\n' + '\n'.join(hp)
