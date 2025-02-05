import pandas as pd
from datetime import datetime

from ssl_eeg import model, preprocessing as pr, preprocessing_nback as prn, train_sl

# current hyperparameter configuration
models_conf = pd.read_csv(model.models_conf_path, index_col=0)
conf_ids = [] # configuration ids to train
use_previous = True # load previously trained model, int or boolean

# training parameters
batch_size = 256
val_batch_size = 2048

################################################################

for cur_conf_id in conf_ids:
    cur_conf = models_conf.loc[cur_conf_id]
    print(cur_conf)

    # data frame containing the filtered 8 eeg-channels, n-value from n-back-task, session number 
    # and block number | data frame for eeg chunks
    blocks_df, chunks = prn.arange_data(filter_data=True, lowpass=cur_conf["filter"], trans_band_auto=False, verbose=False)

    # one session for testing
    train_chunks, test_chunks = prn.get_train_test_sets(chunks, test_session=cur_conf["test_session"])
    chunks_data_X, chunks_data_Y = prn.get_samples_data(train_chunks, blocks_df)
    chunks_data_X = pr.normalize_data(chunks_data_X)

    folds = pr.get_folds(train_chunks, k_folds=10)

    for f in range(cur_conf["to_train"]):
        epochs = cur_conf["epochs"]

        # preparing training and validation data
        v_i = cur_conf["models_trained"]
        train, val = pr.get_train_val_sets(folds, v_i)
        
        # model
        siam_nn = model.SiameseEegNet(out_dim=cur_conf["out_dim"], dropout_p=cur_conf["dropout_p"])
        model_name = train_sl.get_model_name(cur_conf)

        # load previously trained model
        if use_previous:
            if type(use_previous) is int:
                prev_conf = use_previous
            else:
                print("Searching for model with same configuration")
                prev_conf = model.get_same_config(cur_conf_id)

            if prev_conf > 0:
                prev_model = f"conf_id_{prev_conf}_model_" + str(v_i+1)
                print("Loading previous model", prev_model)
                siam_nn = model.load_model(prev_model, out_dim=cur_conf["out_dim"], dropout_p=cur_conf["dropout_p"])
                epochs = epochs - models_conf.loc[prev_conf]["epochs"]
        
        # training
        print(f"Starting training of model {model_name}")
        losses = train_sl.train(siam_nn, epochs=epochs, chunks_data=chunks_data_X, train_chunks_df=train, val_chunks_df=val, batch_size=batch_size, val_batch_size=val_batch_size, learning_rate=cur_conf["learning_rate"], loss_margin=cur_conf["loss_margin"], distance_type=cur_conf["distance_type"], name=model_name)
        print(f"Finished training of model {model_name}")

        # save to model_doc
        train_loss = losses["train_loss"][-1]
        val_loss = min(losses["val_loss"])
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        models_doc = pd.read_csv(model.models_doc_path, index_col=0)
        cur_mo = [cur_conf_id, model_name, now, train_loss, val_loss, epochs, v_i]
        models_doc.loc[models_doc.iloc[-1].name+1] = cur_mo + [0] * (models_doc.shape[1] - len(cur_mo))
        models_doc.to_csv(model.models_doc_path)
        
        # update model_conf
        models_conf = pd.read_csv(model.models_conf_path, index_col=0)
        models_conf.loc[cur_conf_id, "models_trained"] += 1
        models_conf.loc[cur_conf_id, "to_train"] -= 1
        models_conf.to_csv(model.models_conf_path)
        cur_conf = models_conf.loc[cur_conf_id]
