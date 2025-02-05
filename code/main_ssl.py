import pandas as pd
import torch
from datetime import datetime

from ssl_eeg import model, train_ssl, preprocessing as pr, preprocessing_unlabeled as pru, preprocessing_nback as prn

val_sessions = [1,3]

# current hyperparameter configuration
models_conf = pd.read_csv(model.models_conf_path, index_col=0)
conf_ids = [] # configuration ids to train
use_previous = False
train_on_nback = False

# training parameters
batch_size = 256
val_batch_size = 2048

################################################################

for cur_conf_id in conf_ids:
    cur_conf = models_conf.loc[cur_conf_id]
    epochs = cur_conf["epochs"]
    print(cur_conf)

    # UNLABELED DATA
    blocks_df, chunks = pru.arange_data(lowpass=cur_conf["filter"], trans_band_auto=False, verbose=False)
    blocks_t = torch.from_numpy(blocks_df.to_numpy()).T
    val_blocks_t = blocks_t

    if train_on_nback:
        # LABELED N-BACK DATA
        blocks_ndf, chunks_n = prn.arange_data(filter_data=True, lowpass=cur_conf["filter"], trans_band_auto=False, verbose=False)
        blocks_nt = torch.from_numpy(blocks_ndf.drop(columns=["time_stamp", "n"]).to_numpy()).T

        train_chunks_ndf, test_chunks_ndf = prn.get_train_test_sets(chunks, prn.test_ses)
        train_chunks_ndf = train_chunks_ndf.drop(columns=["n", "offset"])
        train_chunks_nt = torch.from_numpy(train_chunks_ndf.to_numpy())

    if cur_conf["base_model"] > 0:
        use_previous = cur_conf["base_model"]
        print(f"Using base model: {cur_conf['base_model']}") 

    for f in range(cur_conf["to_train"]):

        # preparing disjoint training and validation data
        v_i = val_sessions[cur_conf["models_trained"]]
        print("Validation session:", v_i)

        train_chunks = chunks.loc[chunks["session_no"] != v_i]
        val_chunks = chunks.loc[chunks["session_no"] == v_i]
        train_chunks = torch.from_numpy(train_chunks.to_numpy())
        val_chunks = torch.from_numpy(val_chunks.to_numpy())

        if train_on_nback:
            print("Training on labeled n-back data")
            blocks_t = blocks_nt
            train_chunks = train_chunks_nt

        # model
        siam_nn = model.SiameseEegNet(out_dim=cur_conf["out_dim"], dropout_p=cur_conf["dropout_p"])
        model_name = train_ssl.get_model_name(cur_conf, val_sessions)
        
        # load a previously trained model
        if use_previous:
            if type(use_previous) is int:
                prev_conf = use_previous
            else:
                print("Searching for model with same configuration")
                prev_conf = model.get_same_config(cur_conf_id)

            if prev_conf > 0:
                prev_model = f"conf_id_{prev_conf}_model_" + str(v_i) + "_ssl"
                print("Loading previous model", prev_model)
                siam_nn = model.load_model(prev_model, out_dim=cur_conf["out_dim"], dropout_p=cur_conf["dropout_p"])
                epochs = epochs - models_conf.loc[prev_conf]["epochs"]

        # training
        print(f"Starting training of model {model_name}")
        losses = train_ssl.train(siam_nn, epochs=epochs, blocks=blocks_t, train_chunks=train_chunks, val_chunks=val_chunks, batch_size=batch_size, val_batch_size=val_batch_size, learning_rate=cur_conf["learning_rate"], loss_margin=cur_conf["loss_margin"], distance_type=cur_conf["distance_type"], name=model_name, augmentation_scale=cur_conf["augmentation_scale"], val_blocks_tensor=val_blocks_t)
        print(f"Finished training of model {model_name}")

        val_loss = min(losses["val_loss"])
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        models_doc = pd.read_csv(model.models_doc_path, index_col=0)
        cur_mo = [cur_conf_id, model_name, now, losses["train_loss"][-1], val_loss, epochs, v_i]
        models_doc.loc[models_doc.iloc[-1].name+1] = cur_mo + [0] * (models_doc.shape[1] - len(cur_mo))
        models_doc.to_csv(model.models_doc_path)
        
        # update model_conf
        models_conf = pd.read_csv(model.models_conf_path, index_col=0)
        models_conf.loc[cur_conf_id, "models_trained"] += 1
        models_conf.loc[cur_conf_id, "to_train"] -= 1
        models_conf.to_csv(model.models_conf_path)
        cur_conf = models_conf.loc[cur_conf_id]
