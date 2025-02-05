import pandas as pd
import torch
from datetime import datetime

from ssl_eeg import model, preprocessing as pr, preprocessing_nback as prn, prediction as prd, head
from ssl_eeg.train_sl import device, get_model_name

models_conf = pd.read_csv(model.models_conf_path, index_col=0)
models_doc = pd.read_csv(model.models_doc_path, index_col=0)

# current hyperparameter configuration
models_conf_head = pd.read_csv(model.models_conf_head_path, index_col=0)
conf_ids = [] # configuration ids to trainy

# training parameters
batch_size = 256
val_batch_size = 2048

################################################################

for cur_conf_id in conf_ids:
    cur_conf = models_conf_head.loc[cur_conf_id]
    print(cur_conf)

    base_conf = models_conf.loc[cur_conf["base_model_conf"]]
    base_models = models_doc.loc[models_doc["conf_id"]==base_conf.name]

    # data frame containing the filtered 8 eeg-channels, n-value from n-back-task, session number 
    # and block number | data frame for eeg chunks
    blocks_df, chunks = prn.arange_data(filter_data=True, lowpass=base_conf["filter"], trans_band_auto=False, verbose=False)

    # one session for testing
    if base_conf["supervised"] == "supervised":
        test_ses = base_conf["test_session"]
    else:
        test_ses = cur_conf["test_session"]

    print("Using test session", test_ses)
    train_chunks, test_chunks = prn.get_train_test_sets(chunks, test_session=test_ses)
    chunks_data_X, chunks_data_Y = prn.get_samples_data(train_chunks, blocks_df)
    chunks_data_X = pr.normalize_data(chunks_data_X)
    
    folds = pr.get_folds(train_chunks, k_folds=10)

    for f in range(int(cur_conf["to_train"])):

        # preparing training and validation data
        v_i = int(cur_conf["models_trained"])
        train_set, val_set = pr.get_train_val_sets(folds, v_i)

        # base model
        cur_base = base_models.iloc[f]
        base_name = cur_base["model_name"] + "_best_val"
        print("Loading base model", base_name)
        base_model = model.load_model(base_name, base_conf["out_dim"], base_conf["dropout_p"])

        # feature representation of data, output of base model
        train_features, train_y = prd.get_model_output_n(base_model, train_set, blocks_df, as_df=False)
        train_y = train_y.to(torch.long)
        train_y = train_y.to(device)

        val_features, val_y = prd.get_model_output_n(base_model, val_set, blocks_df, as_df=False)
        val_y = val_y.to(torch.long)
        val_y = val_y.to(device)

        # head model
        head_name = "head_" + get_model_name(cur_conf)
        head_model = model.HeadNet(in_dim=base_conf["out_dim"])
        epochs = int(cur_conf["epochs"])

        print(f"Starting training of model {head_name}")
        losses = head.train(head_model, epochs, train_features, train_y, val_features, val_y, batch_size, val_batch_size, head_name, cur_conf["learning_rate"])
        print(f"Finished training of model {head_name}")

        # save to model_doc
        train_loss = losses["train_loss"][-1]
        val_loss = min(losses["val_loss"])
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        models_doc_head = pd.read_csv(model.models_doc_head_path, index_col=0)
        cur_mo = [cur_conf_id, cur_base.name, head_name, now, train_loss, val_loss, epochs, v_i]
        models_doc_head.loc[models_doc_head.iloc[-1].name+1] = cur_mo + [0] * (models_doc_head.shape[1] - len(cur_mo))
        models_doc_head.to_csv(model.models_doc_head_path)
        
        # update model_conf
        models_conf_head = pd.read_csv(model.models_conf_head_path, index_col=0)
        models_conf_head.loc[cur_conf_id, "models_trained"] += 1
        models_conf_head.loc[cur_conf_id, "to_train"] -= 1
        models_conf_head.to_csv(model.models_conf_head_path)
        cur_conf = models_conf_head.loc[cur_conf_id]