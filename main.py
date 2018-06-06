#  code:utf-8
import io
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from preprocessing import label_encode
from preprocessing import text_process
from preprocessing import get_keras_data

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def get_model():
    dr_r = 0.1

    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    emb_name = Embedding(MAX_TEXT, 50)(name)
    emb_item_desc = Embedding(MAX_TEXT, 50)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)

    main_l = concatenate([Flatten() (emb_brand_name)
                          , Flatten() (emb_category_name)
                          , Flatten() (emb_item_condition)
                          , rnn_layer1
                          , rnn_layer2
                          , num_vars])

    main_l = Dropout(dr_r) (Dense(128) (main_l))
    main_l = Dropout(dr_r) (Dense(64) (main_l))

    output = Dense(1, activation="linear") (main_l)
    model = Model([name, item_desc, brand_name
                   , category_name, item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])

    return model


def RMSLE(y,y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(np.log(y_pred[i] + 1) - np.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5


def draw_heatmap(data, row_labels, column_labels):
    fig, ax = plt.subplots(figsize=(12,10))
    heatmap = ax.pcolor(data, cmap=plt.cm.tab20c)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels(row_labels, minor=False,rotation=90, fontsize=10)
    ax.set_yticklabels(column_labels, minor=False,rotation=0, fontsize=8)

    plt.savefig('image.png')
    return heatmap


def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return dataset


if __name__ == '__main__':
    train = pd.read_csv('train.tsv', delimiter='\t')
    test = pd.read_csv('test.tsv', delimiter='\t')

    train = handle_missing(train)
    test = handle_missing(test)

    train, test = label_encode(train, test)
    train, test = text_process(train, test)

    max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))),
                           np.max(test.seq_name.apply(lambda x: len(x)))])

    max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x)))
                             , np.max(test.seq_item_description.apply(lambda x: len(x)))])

    print("max name seq "+str(max_name_seq))
    print("max item desc seq "+str(max_seq_item_description))

    MAX_NAME_SEQ = 10
    MAX_ITEM_DESC_SEQ = 75

    MAX_TEXT = np.max([np.max(train.seq_name.max())
                      ,np.max(test.seq_name.max())
                      ,np.max(train.seq_item_description.max())
                      ,np.max(test.seq_item_description.max())])+2
    MAX_CATEGORY = np.max([train.category_name.max(), test.category_name.max()])+1
    MAX_BRAND = np.max([train.brand_name.max(), test.brand_name.max()])+1
    MAX_CONDITION = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1

    train["target"] = np.log(train.price+1)
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    train["target"] = target_scaler.fit_transform(train.target.reshape(-1,1))

    dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.99)
    print(dtrain.shape)
    print(dvalid.shape)

    X_train = get_keras_data(dtrain,MAX_NAME_SEQ,MAX_ITEM_DESC_SEQ)
    X_valid = get_keras_data(dvalid,MAX_NAME_SEQ,MAX_ITEM_DESC_SEQ)
    X_test = get_keras_data(test,MAX_NAME_SEQ,MAX_ITEM_DESC_SEQ)

    model = get_model()
    model.summary()

    BATCH_SIZE = 20000
    epochs = 5

    model.fit(X_train, dtrain.target, epochs=epochs, batch_size=BATCH_SIZE
          , validation_data=(X_valid, dvalid.target)
          , verbose=1)

    val_preds = model.predict(X_valid)
    val_preds = target_scaler.inverse_transform(val_preds)
    val_preds = np.exp(val_preds)+1

    #mean_absolute_error, mean_squared_log_error
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:,0]
    v_rmsle = RMSLE(y_true, y_pred)
    print(" RMSLE error on dev test: "+str(v_rmsle))

    '''
    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    preds = target_scaler.inverse_transform(preds)
    preds = np.exp(preds)-1

    submission = test[["test_id"]]
    submission["price"] = preds
    '''

    #print(df.corr().head())
    #heatmap = draw_heatmap(df.corr(),df.corr().index,df.corr().columns)
    '''
    sentences = item_description_tokenizer(df['item_description'])
    weighted_matrix , vector_name = description_vectorizer(sentences)

    weighted_matrix.index = df.index
    weighted_matrix.columns = vector_name

    dfp = pd.concat([df['price'] , weighted_matrix],axis = 1)


    target = dfp['price'].values.astype(np.int64)
    del dfp['price']
    #del dfp['item_description']
    data_names = dfp.columns


    clf = linear_model.LinearRegression()
    X = dfp.values.astype(np.int64)
    Y = target
    clf.fit(X, Y)


    print(pd.DataFrame({"Name":data_names,"Coefficients":clf.coef_}).sort_values(by='Coefficients') )


    y_true = target
    y_pred = clf.predict(X)
    res = RMSLE(y_true,y_pred)
    print(res)





    target = df['price'].values.astype(np.int64)
    del df['price']
    del df['item_description']
    del df['name']
    data_names = df.columns


    clf = linear_model.LinearRegression()
    X = df.values.astype(np.int64)
    Y = target
    clf.fit(X, Y)


    print(pd.DataFrame({"Name":data_names,"Coefficients":clf.coef_}).sort_values(by='Coefficients') )


    y_true = target
    y_pred = clf.predict(X)
    res = RMSLE(y_true,y_pred)
    print(res)
    '''
